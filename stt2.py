#!/usr/bin/env python3
# stt_sr_to_topic.py
# ROS 2 node: microphone -> speech_recognition -> /voice/text
#
# Env (optional):
#   VOICE_TOPIC=/voice/text
#   MIC_INDEX=default         # integer device index, or leave as default
#   LANG=en-GB                # Google language code (e.g., en-US, en-GB)
#   PHRASE_TIME_LIMIT=8       # max seconds per utterance (int)
#   DYNAMIC_ENERGY=1          # 1 to auto-adapt to noise
#   ENERGY_THRESHOLD=300      # manual threshold if DYNAMIC_ENERGY=0

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import speech_recognition as sr

def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "")
    if not v:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

class SRPublisher(Node):
    def __init__(self,
                 topic: str = "/voice/text",
                 mic_index=None,
                 language: str = "en-GB",
                 phrase_time_limit: int | None = 10,
                 dynamic_energy: bool = True,
                 energy_threshold: int | None = None):
        super().__init__("sr_to_topic")
        self.pub = self.create_publisher(String, topic, 10)
        self.get_logger().info(f"Publishing recognised text to: {topic}")
        self.get_logger().info(f"Mic index: {mic_index if mic_index is not None else 'default'} | Lang: {language}")

        # recognizer config
        self.r = sr.Recognizer()
        self.r.dynamic_energy_threshold = dynamic_energy
        if not dynamic_energy and energy_threshold is not None:
            self.r.energy_threshold = energy_threshold

        self.language = language
        self.phrase_time_limit = phrase_time_limit

        # microphone + ambient calibration
        self.mic = sr.Microphone(device_index=mic_index)
        with self.mic as source:
            self.get_logger().info("Calibrating for ambient noise…")
            self.r.adjust_for_ambient_noise(source, duration=1.0)
            self.get_logger().info(f"Calibrated. Energy threshold: {self.r.energy_threshold:.1f}")

        # start background listener
        self.stopper = self.r.listen_in_background(
            self.mic,
            self._callback,
            phrase_time_limit=self.phrase_time_limit
        )
        self.get_logger().info("Listening… speak a phrase and pause.")

    def _callback(self, recognizer: sr.Recognizer, audio: sr.AudioData):
        text = ""
        try:
            # Google Web Speech (no API key needed for basic usage)
            text = recognizer.recognize_google(audio, language=self.language).strip()
        except sr.UnknownValueError:
            text = ""  # ignore unintelligible audio
        except sr.RequestError as e:
            self.get_logger().error(f"Speech API error: {e}")
            text = "[asr_error] " + str(e)

        if text:
            msg = String(); msg.data = text
            self.pub.publish(msg)
            self.get_logger().info(f"→ {text}")

    def shutdown(self):
        try:
            if self.stopper:
                # stop background thread (do not block)
                self.stopper(wait_for_stop=False)
        except Exception:
            pass

def main():
    rclpy.init()
    topic = os.environ.get("VOICE_TOPIC", "/voice/text")
    lang = os.environ.get("LANG", "en-GB")

    mic_env = os.environ.get("MIC_INDEX", "").strip().lower()
    if mic_env in ("", "default", "none"):
        mic_index = None
    else:
        try:
            mic_index = int(mic_env)
        except ValueError:
            mic_index = None

    ptl_env = os.environ.get("PHRASE_TIME_LIMIT", "").strip()
    phrase_time_limit = int(ptl_env) if ptl_env.isdigit() else 8

    dynamic_energy = _env_bool("DYNAMIC_ENERGY", True)
    eth_env = os.environ.get("ENERGY_THRESHOLD", "").strip()
    energy_threshold = int(eth_env) if (eth_env.isdigit() and not dynamic_energy) else None

    node = SRPublisher(topic, mic_index, lang, phrase_time_limit, dynamic_energy, energy_threshold)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
