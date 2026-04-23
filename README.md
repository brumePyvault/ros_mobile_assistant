# Voice-Assisted TurtleBot3 (Waffle Pi) with Local LLM

Natural-language control for TurtleBot3 Waffle Pi using a local LLM (Ollama), speech recognition, and Nav2. Speak commands like “go to the kitchen,” “save waypoint door here,” “come back,” “turn left,” or “move forward for two seconds,” and the robot will act.

This project is split across two machines:

* **SBC**: Raspberry Pi 4 running Ubuntu 22.04.5 Server LTS, connected to OpenCR, LDS-02, and a camera, with ROS 2 installed.
* **Remote PC**: Ubuntu 22.04 Desktop with 16 GB RAM and GTX 1070, microphone attached, running the LLM and speech-to-text.

> Make sure your TurtleBot3 model is Waffle Pi. Set `TURTLEBOT3_MODEL=waffle_pi` everywhere.

---

## Features

* Voice commands to navigate to saved waypoints, save new waypoints, come back, stop, ask weather, or drive manually.
* Uses Ollama locally for intent parsing into a simple JSON schema.
* Speech recognition to publish recognised phrases as `/voice/text`.
* Optional local TTS on the Remote PC with `pyttsx3`.
* Safe manual driving with rate-controlled `/cmd_vel` and configurable caps.
* Waypoint persistence to a JSON file.
* Works alongside Nav2 on your existing map.

---

## Repository Layout

```
.
├── voice_assistance.py     # Core assistant node (Remote PC or SBC; designed for Remote PC)
├── stt.py      # ElevenLabs STT -> /voice/text publisher (Remote PC)
├── waypoints.json          # Created on first save; holds named poses
└── README.md
```

---

## Requirements

**Both machines**

* ROS 2 Humble
* Network connectivity on the same LAN and the same `ROS_DOMAIN_ID`
* `export TURTLEBOT3_MODEL=waffle_pi`

**SBC (Pi 4)**

* TurtleBot3 Waffle Pi hardware with OpenCR and LDS-02
* `turtlebot3_bringup`, Nav2 stack
* OpenCR connected at `/dev/ttyACM0` (or a udev symlink `/dev/opencr`)

**Remote PC**

* Python 3.10
* `pip install requests pyttsx3 speechrecognition`
* **Ollama** running locally with a model pulled, e.g. `llama3.1:8b`

---

## Installation

### 1) SBC: Bring the robot up

```bash
# On SBC
echo 'export TURTLEBOT3_MODEL=waffle_pi' >> ~/.bashrc
source ~/.bashrc

# Optional udev rule for a stable port name
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", SYMLINK+="opencr", MODE="0666"' | sudo tee /etc/udev/rules.d/99-opencr.rules
sudo udevadm control --reload && sudo udevadm trigger

# Launch TurtleBot3 bringup
source /opt/ros/humble/setup.bash
ros2 launch turtlebot3_bringup robot.launch.py port:=/dev/opencr
```

You should see topics like `/cmd_vel`, `/odom`, `/scan`, `/imu`, `/battery_state`.

### 2) Remote PC: Start Ollama and install deps

```bash
# Ollama
ollama serve
ollama pull llama3.1:8b

# Python deps
python3 -m pip install --user requests pyttsx3 SpeechRecognition python-dotenv
```

---

## Running

Open two terminals on the **Remote PC**.

### A) Start speech-to-text publisher

This waits for a keypress to start recording from your microphone, then another keypress to stop and publish recognised text to `/voice/text`.

```bash
source /opt/ros/humble/setup.bash
export VOICE_TOPIC=/voice/text
export ELEVENLABS_API_KEY=<your_key>
export ELEVENLABS_STT_MODEL_ID=scribe_v1
export VOICE_LANG=en          # e.g. en
# Optional: keep ALSA warnings quiet on Linux (default true)
export SUPPRESS_ALSA_ERRORS=true
python3 stt.py
# Press any key to start recording, press any key again to stop/transcribe
```

### B) Start the assistant

This consumes `/voice/text`, calls the LLM, then navigates, saves waypoints, drives, or answers weather.

```bash
source /opt/ros/humble/setup.bash
python3 voice_assistance.py --waypoints ./waypoints.json --city Leicester
```

Optional environment variables:

* `LLM_URL` (default `http://127.0.0.1:11434`)
* `LLM_MODEL` (default `llama3.1:8b`)
* `CITY` default fallback for weather
* `--speak-local true|false` to enable laptop TTS

---

## Voice Commands You Can Say

* “Go to the **kitchen**” → Navigate to saved waypoint `kitchen`.
* “Save waypoint **door** here” → Save current pose as `door`.
* “Come back” or “Return” → Go back to the start of current goal or `home`.
* “Stop” or “Cancel” → Cancel navigation and stop the robot.
* “Turn left for two seconds” → Manual spin with timed `/cmd_vel`.
* “Move forward for three seconds” → Timed forward drive.
* “What is the weather in **Leicester**?” → Weather summary via wttr.in.

> The assistant always produces a JSON intent internally, then acts.

---

## Topics and Actions

**Subscribed**

* `/voice/text` (`std_msgs/String`) from `stt.py`
* `/amcl_pose` (`geometry_msgs/PoseWithCovarianceStamped`) for pose memory

**Published**

* `/assistant/say` (`std_msgs/String`) optional text to speak
* `/assistant/status` (`std_msgs/String`) brief status messages
* `/cmd_vel` (`geometry_msgs/Twist`) when performing manual drive

**Actions**

* `navigate_to_pose` (`nav2_msgs/action/NavigateToPose`) for Nav2 goals

---

## Waypoints

Saved to `waypoints.json` as simple map poses:

```json
{
  "home": {"x": 0.12, "y": -1.03, "yaw_deg": 180.0},
  "door": {"x": -2.0, "y": 0.5, "yaw_deg": 90.0}
}
```

Names are normalised to lower case, safe characters, and single spaces.

---

## Configuration

Manual drive speed caps are set inside `voice_assistance.py`:

```python
DRIVE_HZ = 20.0
MAX_LIN = 0.30   # m/s
MAX_ANG = 1.2    # rad/s
```

Waffle Pi is factory limited to about **0.26 m/s**. You can lift Nav2 and teleop limits, but OpenCR may still clamp above \~0.26. Increase with care and only if you know what you are doing.

---

## Troubleshooting

**Robot beeps and will not move**

* Ensure `TURTLEBOT3_MODEL=waffle_pi` on both machines. Using `waffle` loads the wrong profile and can trip faults.
* Re-enable torque after a fault:

  ```bash
  ros2 service call /motor_power std_srvs/srv/SetBool "{data: true}"
  ```
* Check power:

  ```bash
  ros2 topic echo /battery_state
  ```

  If voltage sags near \~11 V under load, recharge.
* Confirm OpenCR is present: `ls /dev/ttyACM*` or `/dev/opencr`. If missing, reseat the USB cable and relaunch bringup.

**Nav2 is active but nothing moves**

* Make sure you are not sending only angular or only linear zeros.
* If sending `/cmd_vel` manually from the Remote PC, verify it is visible on the SBC and that no velocity smoother is clamping it.

**Max speed is 0.26 m/s**

* This is expected for Waffle Pi. You may raise Nav2 controller limits, but OpenCR still clamps near 0.26 unless you modify firmware. Faster speeds increase current draw and braking distance.

**Speech recognition does not hear you**

* Change `MIC_INDEX`, calibrate ambient noise, and ensure `LANG` matches your accent. The script logs errors from ElevenLabs STT API if unavailable.

---

## Security Notes

* Ollama is bound to localhost by default. Do not expose it on an untrusted network.
* This project sends only short prompts to the LLM and never publishes camera or map data.

---

## Development Tips

* Test commands without your voice:

  ```bash
  ros2 topic pub /voice/text std_msgs/String "data: 'save waypoint door here'"
  ros2 topic pub /voice/text std_msgs/String "data: 'go to door'"
  ros2 topic pub /voice/text std_msgs/String "data: 'turn left for two seconds'"
  ```
* Inspect intents by running the assistant with `--ros-args --log-level debug` to see LLM outputs.

---

## Roadmap

* Add small safety layer that refuses forward motion when the laser sees a near obstacle.
* Add camera-based person following and object search intents.
* Replace Google Web Speech with Whisper for fully offline STT.

---

## Licence

MIT Licence. See `LICENSE` if provided, otherwise treat this repository as MIT.

---

## Acknowledgements

* TurtleBot3 packages by ROBOTIS.
* Nav2 stack.
* Ollama for local LLM inference.
