#!/usr/bin/env python3
# voice_assistance.py
#
# Remote PC: subscribes to /voice/text -> asks local LLM (Ollama) for intent JSON -> acts (Nav2 / save waypoint / weather / drive)
# Deps: pip install requests pyttsx3
# Start Ollama separately (default http://127.0.0.1:11434), e.g.:
#   ollama serve
#   ollama pull llama3.1:8b
#
# Run:
#   source /opt/ros/humble/setup.bash
#   python voice_assistance.py --waypoints ./waypoints.json --city London
#
# Topics:
#   Sub: /voice/text (std_msgs/String)       — your mic→STT publisher sends here
#   Pub: /assistant/say (std_msgs/String)    — text to speak (robot or laptop)
#   Pub: /cmd_vel (geometry_msgs/Twist)      — manual DRIVE and STOP pings
#   Action: navigate_to_pose (nav2_msgs/NavigateToPose)

import os
import re
import json
import math
import argparse
import threading
import queue
import atexit
from typing import Optional, Dict, Any

import requests

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion, Twist, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose

from tf2_ros import Buffer, TransformListener

# ===================== TTS (robust, background thread) =====================

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

_tts_q: "queue.Queue[Optional[str]]" = queue.Queue()
_tts_thread_started = False
_tts_engine = None

def _tts_worker():
    global _tts_engine
    if pyttsx3 is None:
        return
    _tts_engine = pyttsx3.init()
    try:
        while True:
            item = _tts_q.get()
            if item is None:
                break
            text = (item or "").strip()
            if not text:
                continue
            _tts_engine.say(text)
            _tts_engine.runAndWait()
    finally:
        try:
            _tts_engine.stop()
        except Exception:
            pass

def _ensure_tts_thread():
    global _tts_thread_started
    if pyttsx3 is None:
        return
    if not _tts_thread_started:
        t = threading.Thread(target=_tts_worker, daemon=True)
        t.start()
        atexit.register(lambda: _tts_q.put(None))
        _tts_thread_started = True

def speak_local(text: str, enabled: bool):
    if not enabled or not text:
        return
    _ensure_tts_thread()
    _tts_q.put(text)

# ===================== Small utilities =====================

def yaw_to_quat(yaw_deg: float) -> Quaternion:
    yaw = math.radians(yaw_deg)
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw/2.0), w=math.cos(yaw/2.0))

def quat_to_yaw_deg(q: Quaternion) -> float:
    # Planar yaw only
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))

def load_waypoints(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_waypoints(path: str, data: dict):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def norm_name(name: str) -> str:
    # normalize waypoint names: lowercase, single spaces, safe chars
    n = re.sub(r'[^a-z0-9_-]+', ' ', (name or "").lower()).strip()
    return re.sub(r'\s+', ' ', n)

def get_weather_text(default_city: str, override_city: Optional[str]):
    city = (override_city or default_city or "London").strip()
    try:
        r = requests.get(f"https://wttr.in/{city}?format=j1", timeout=6)
        c = r.json()["current_condition"][0]
        desc = c["weatherDesc"][0]["value"]
        temp = c["temp_C"]; feels = c["FeelsLikeC"]
        return f"The weather in {city} is {desc}, {temp} degrees, feels like {feels}."
    except Exception:
        return f"I could not fetch the weather for {city} right now."

# ===================== Manual DRIVE settings =====================

DRIVE_HZ = 20.0
MAX_LIN = 0.30   # m/s — safe-ish for TurtleBot
MAX_ANG = 1.2    # rad/s

# ===================== LLM plumbing (Ollama by default) =====================

LLM_SYSTEM_PROMPT = """You are the brain of a mobile robot that lives on a saved 2D map.
You ALWAYS reply with ONE valid JSON object, no commentary.
Schema:
{
  "intent": "NAVIGATE" | "STOP" | "COME_BACK" | "SAVE_WAYPOINT" | "WEATHER" | "DRIVE" | "SMALL_TALK" | "OTHER",
  "slots": {
    "location": "<saved waypoint name if navigating>",
    "waypoint_name": "<name to save current position under>",
    "city": "<city for weather>",
    "linear_x": "<m/s forward (+) or backward (-) for DRIVE> remember: -0.26 >= x <= 0.26 (use full range mostly both directions)",
    "angular_z": "<rad/s left (+) or right (-) for DRIVE>",
    "duration_s": "<seconds to apply the DRIVE command>"
  },
  "say": "<one short sentence to speak back>"
}
Rules:
- Free phrasing like "I want you at the door" -> NAVIGATE to 'door' if it matches a known waypoint.
- "turn left/right" -> DRIVE with angular_z positive(left)/negative(right), linear_x=0, duration ≈1–2 s.
- "move forward/back" -> DRIVE with linear_x positive(forward)/negative(backwards), angular_z=0, duration ≈1–3 s.
- "moves likes spin and drive like a snake make  up something from this ' "turn left/right" -> DRIVE with angular_z positive(left)/negative(right), linear_x=0, duration ≈ \{make your time\}s. "move forward/back" -> DRIVE with linear_x positive(forward)/negative(backwards), angular_z=0,duration ≈ \{make your time\} s.'"
- Robot is not holonomic; "move left/right" should be interpreted as a turn, not a strafe.
- "stop", "cancel", "wait" -> STOP.
- "come back", "return", "go back" -> COME_BACK.
- "save waypoint <name> here"/"remember <name>" -> SAVE_WAYPOINT.
- Weather questions -> WEATHER (fill 'city' if user said one).
- If none of the above, use SMALL_TALK and fill 'say'.
- Keep 'say' short. Output ONLY the JSON object. No code fences, no commentary.
"""

def call_ollama_chat(prompt: str, url: str, model: str, timeout=25) -> str:
    """Return assistant text (one non-streaming chat response)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0}
    }
    try:
        resp = requests.post(url.rstrip("/") + "/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        if "response" in data:
            return data["response"]
        if "messages" in data and data["messages"]:
            return data["messages"][-1].get("content", "")
    except Exception as e:
        print("[ollama] error:", e)
    # Fallback small-talk JSON
    return json.dumps({"intent": "OTHER", "slots": {}, "say": "I had trouble thinking just now."})

def extract_first_json(s: str) -> dict:
    """Extract first top-level JSON object from s."""
    in_str = False
    esc = False
    depth = 0
    start = -1
    for i, ch in enumerate(s or ""):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        chunk = s[start:i+1]
                        try:
                            return json.loads(chunk)
                        except Exception:
                            start = -1
    return {"intent": "SMALL_TALK", "slots": {}, "say": (s or "").strip()[:180]}

# ===================== Main ROS2 node =====================

class AssistantCoreLLM(Node):
    def __init__(self, waypoints_path: str, map_frame: str, city: str, speak_here: bool,
                 llm_url: str, llm_model: str):
        super().__init__('assistant_core_llm')

        # config
        self.map_frame = map_frame
        self.city = city
        self.speak_here = speak_here
        self.waypoints_path = waypoints_path
        self.waypoints = load_waypoints(waypoints_path)

        # LLM backend
        self.llm_url = llm_url
        self.llm_model = llm_model

        # pubs/subs
        self.say_pub = self.create_publisher(String, "/assistant/say", 10)
        self.status_pub = self.create_publisher(String, "/assistant/status", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(String, "/voice/text", self.on_text, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.on_pose, 10)

        # TF as a fallback if /amcl_pose not present yet
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.2, self._update_pose_from_tf)  # 5 Hz

        # Nav2
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.current_goal = None

        # Pose memory for "come back"
        self.latest_pose: Optional[Dict[str, float]] = None
        self.home_pose = self.waypoints.get("home")
        self.start_pose_of_current_goal = None

        # Manual drive state
        self._drive_active = False
        self._drive_until = 0.0
        self._drive_linear = 0.0
        self._drive_angular = 0.0
        self._drive_timer = self.create_timer(1.0/DRIVE_HZ, self._drive_tick)

        self.get_logger().info("AssistantCoreLLM ready. Listening to /voice/text.")

    # ------- helpers -------
    def say(self, text: str):
        if not text:
            return
        self.say_pub.publish(String(data=text))
        speak_local(text, self.speak_here)

    def on_pose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose
        pose = {
            "x": float(p.position.x),
            "y": float(p.position.y),
            "yaw_deg": float(quat_to_yaw_deg(p.orientation))
        }
        self.latest_pose = pose
        if self.home_pose is None:
            self.home_pose = pose

    def _update_pose_from_tf(self):
        # If AMCL hasn't set a pose yet, try TF lookup map->base_link
        if self.latest_pose is not None:
            return
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, 'base_link', Time())
            x = float(t.transform.translation.x)
            y = float(t.transform.translation.y)
            q = t.transform.rotation
            yaw = quat_to_yaw_deg(q)
            self.latest_pose = {"x": x, "y": y, "yaw_deg": yaw}
        except Exception:
            pass

    # ------- message handling -------
    def on_text(self, msg: String):
        user_text = msg.data.strip()
        if not user_text:
            return
        lt = user_text.lower()

        # Safety fast-paths: handle WITHOUT LLM delay
        if re.search(r"\b(stop|cancel|halt|wait)\b", lt):
            self.cancel_nav()
            self.cmd_vel_pub.publish(Twist())
            self._drive_active = False
            self.say("Stopped.")
            return
        if re.search(r"\b(come back|go back|return)\b", lt):
            self._come_back()
            return

        # Help the model by listing known waypoints
        wp_list = ", ".join(sorted(self.waypoints.keys())) if self.waypoints else "none"
        prompt = (
            f"Waypoints you know about: {wp_list}\n"
            f'User said: "{user_text}"\n'
            "Remember: output ONLY ONE JSON object following the schema."
        )

        raw = call_ollama_chat(prompt, self.llm_url, self.llm_model)
        self.get_logger().debug("LLM raw: " + (raw or "")[:280].replace("\n", " "))
        data = extract_first_json(raw)

        intent = (data.get("intent") or "").upper()
        slots = data.get("slots") or {}
        say_text = data.get("say") or ""
        self.get_logger().info(f"Decision intent={intent} slots={slots} say={say_text!r}")

        if say_text:
            self.say(say_text)

        if intent == "NAVIGATE":
            self._navigate_to(norm_name(slots.get("location") or "")); return
        if intent == "STOP":
            self.cancel_nav(); self.cmd_vel_pub.publish(Twist()); self._drive_active = False; return
        if intent == "COME_BACK":
            self._come_back(); return
        if intent == "SAVE_WAYPOINT":
            self._save_waypoint(norm_name(slots.get("waypoint_name") or "")); return
        if intent == "WEATHER":
            city = (slots.get("city") or "").strip() or None
            self.say(get_weather_text(self.city, city)); return
        if intent == "DRIVE":
            self._route_drive(slots); return
        # SMALL_TALK / OTHER already spoken

    # ------- actions -------
    def _save_waypoint(self, name: str):
        if not name:
            self.say("Please say a name to save this position.")
            return

        # If we don't have a pose yet, try a one-shot TF lookup now
        if not self.latest_pose:
            try:
                t = self.tf_buffer.lookup_transform(self.map_frame, 'base_link', Time())
                x = float(t.transform.translation.x)
                y = float(t.transform.translation.y)
                q = t.transform.rotation
                yaw = quat_to_yaw_deg(q)
                self.latest_pose = {"x": x, "y": y, "yaw_deg": yaw}
            except Exception:
                pass

        if not self.latest_pose:
            self.say("I do not know my position yet.")
            return

        self.waypoints[name] = self.latest_pose
        if "home" not in self.waypoints and self.home_pose:
            self.waypoints["home"] = self.home_pose
        save_waypoints(self.waypoints_path, self.waypoints)
        self.say(f"Saved waypoint {name}.")

    def _navigate_to(self, location_name: str):
        if not location_name:
            self.say("Which location should I go to?")
            return
        if location_name not in self.waypoints:
            self.say(f"I don't have a waypoint named {location_name}. Say 'save waypoint {location_name} here' first.")
            return
        if self.latest_pose:
            self.start_pose_of_current_goal = self.latest_pose
        else:
            self.start_pose_of_current_goal = self.waypoints.get("home")

        wp = self.waypoints[location_name]
        self._start_navigation(wp["x"], wp["y"], wp.get("yaw_deg", 0.0))

    def _come_back(self):
        target = self.start_pose_of_current_goal or self.waypoints.get("home") or self.latest_pose
        if not target:
            self.say("I do not know where to return to.")
            return
        self._start_navigation(target["x"], target["y"], target.get("yaw_deg", 0.0))
        self.say("Coming back.")

    def _start_navigation(self, x: float, y: float, yaw_deg: float):
        # Cancel manual drive if active
        if self._drive_active:
            self._drive_active = False
            self.cmd_vel_pub.publish(Twist())

        if not self.nav_client.wait_for_server(timeout_sec=0.8):
            self.say("Navigation server is not available.")
            return
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation = yaw_to_quat(float(yaw_deg))

        fut = self.nav_client.send_goal_async(goal, feedback_callback=self._on_nav_feedback)
        fut.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        handle = future.result()
        if not handle or not handle.accepted:
            self.say("Goal was rejected.")
            self.current_goal = None
            return
        self.current_goal = handle
        handle.get_result_async().add_done_callback(self._on_nav_result)

    def _on_nav_feedback(self, fb):
        # could read fb.feedback for distance/ETA if you want
        pass

    def _on_nav_result(self, future):
        self.current_goal = None
        self.say("Arrived.")

    def cancel_nav(self):
        # stop Nav2 goal
        if self.current_goal is not None:
            try:
                self.current_goal.cancel_goal_async()
            except Exception:
                pass
            self.current_goal = None
        # stop manual drive
        if self._drive_active:
            self._drive_active = False
            self.cmd_vel_pub.publish(Twist())

    # ------- DRIVE -------
    def _drive_tick(self):
        if not self._drive_active:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if now >= self._drive_until:
            self._drive_active = False
            self.cmd_vel_pub.publish(Twist())  # stop
            return
        tw = Twist()
        tw.linear.x = self._drive_linear
        tw.angular.z = self._drive_angular
        self.cmd_vel_pub.publish(tw)

    def _start_drive(self, linear_x: float, angular_z: float, duration_s: float):
        # Cancel Nav2 so teleop takes over cleanly
        self.cancel_nav()

        # Clamp to safe caps
        try:
            lin = float(linear_x)
            ang = float(angular_z)
            dur = float(duration_s)
        except Exception:
            lin, ang, dur = 0.0, 0.0, 1.0

        lin = max(-MAX_LIN, min(MAX_LIN, lin))
        ang = max(-MAX_ANG, min(MAX_ANG, ang))
        dur = max(0.1, min(10.0, dur if dur > 0 else 1.0))

        self._drive_linear = lin
        self._drive_angular = ang
        now = self.get_clock().now().nanoseconds / 1e9
        self._drive_until = now + dur
        self._drive_active = True

        direction = "forward" if lin > 0.05 else "backward" if lin < -0.05 else "in place"
        if ang > 0.05: direction = "turning left"
        if ang < -0.05: direction = "turning right"
        self.say(f"Driving {direction} for {dur:.1f} seconds.")

    def _route_drive(self, slots: dict):
        # expect strings from LLM, coerce to floats
        lin = float(str(slots.get("linear_x", 0) or 0))
        ang = float(str(slots.get("angular_z", 0) or 0))
        dur = float(str(slots.get("duration_s", 1) or 1))
        self._start_drive(lin, ang, dur)

# ===================== Entry point =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waypoints", default="./waypoints.json")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--city", default=os.environ.get("CITY", "Leicester"))
    parser.add_argument("--speak-local", default="true", choices=["true", "false"])
    parser.add_argument("--llm-url", default=os.environ.get("LLM_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "llama3.1:8b"))
    args = parser.parse_args()

    rclpy.init()
    node = AssistantCoreLLM(
        waypoints_path=args.waypoints,
        map_frame=args.map_frame,
        city=args.city,
        speak_here=(args.speak_local.lower() == "true"),
        llm_url=args.llm_url,
        llm_model=args.llm_model,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
