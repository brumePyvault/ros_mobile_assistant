#!/usr/bin/env python3

import io
import os
import sys
import termios
import threading
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from dotenv import load_dotenv
import requests
import speech_recognition as sr

load_dotenv()  # take environment variables from .env.


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "")
    if not v:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class ElevenLabsSRPublisher(Node):
    def __init__(
        self,
        topic: str = "/voice/text",
        language: str = "en",
        publish_partials: bool = False,
    ):
        super().__init__("elevenlabs_sr_to_topic")
        self.pub = self.create_publisher(String, topic, 10)

        self.api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("Missing ELEVENLABS_API_KEY in environment.")

        self.model_id = os.environ.get("ELEVENLABS_STT_MODEL_ID", "scribe_v1").strip() or "scribe_v1"
        self.language = language
        self.dynamic_energy = _env_bool("DYNAMIC_ENERGY", True)
        self.energy_threshold = int(os.environ.get("ENERGY_THRESHOLD", "250"))
        self.adjust_for_ambient = _env_bool("ADJUST_FOR_AMBIENT", True)

        # Not currently used by ElevenLabs HTTP STT endpoint, kept for parity/config compatibility.
        self.publish_partials = publish_partials

        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = self.dynamic_energy
        self.recognizer.energy_threshold = self.energy_threshold

        self.microphone = sr.Microphone()
        with self.microphone as source:
            if self.adjust_for_ambient:
                self.get_logger().info("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)

        self._busy = False
        self._stop_event = threading.Event()
        self._keyboard_thread = threading.Thread(target=self._keyboard_record_loop, daemon=True)
        self._keyboard_thread.start()

        self.get_logger().info(f"ElevenLabs STT model={self.model_id} | lang={self.language}")
        self.get_logger().info(f"Publishing recognised text to: {topic}")
        self.get_logger().info("Press any key to START recording, then press any key again to STOP and transcribe.")

    def _publish(self, text: str, tag: str = ""):
        text = text.strip()
        if not text:
            return
        msg = String()
        msg.data = text if not tag else f"[{tag}] {text}"
        self.pub.publish(msg)
        self.get_logger().info(f"→ {msg.data}")

    def _transcribe_with_elevenlabs(self, wav_bytes: bytes) -> str:
        files = {
            "file": ("speech.wav", io.BytesIO(wav_bytes), "audio/wav"),
        }
        data = {
            "model_id": self.model_id,
            "language_code": self.language,
        }
        headers = {"xi-api-key": self.api_key}

        r = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers=headers,
            files=files,
            data=data,
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        return (payload.get("text") or "").strip()

    def _read_single_key(self):
        if not sys.stdin.isatty():
            raise RuntimeError("stdin is not a TTY. Run this script from an interactive terminal.")

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _record_until_keypress(self) -> sr.AudioData:
        with self.microphone as source:
            frames = []
            stop_recording = threading.Event()

            def wait_for_stop_key():
                self._read_single_key()
                stop_recording.set()

            stopper = threading.Thread(target=wait_for_stop_key, daemon=True)
            stopper.start()

            while not stop_recording.is_set() and not self._stop_event.is_set():
                data = source.stream.read(source.CHUNK, exception_on_overflow=False)
                if data:
                    frames.append(data)

        audio_bytes = b"".join(frames)
        if not audio_bytes:
            return sr.AudioData(b"", self.microphone.SAMPLE_RATE, self.microphone.SAMPLE_WIDTH)

        return sr.AudioData(audio_bytes, self.microphone.SAMPLE_RATE, self.microphone.SAMPLE_WIDTH)

    def _keyboard_record_loop(self):
        try:
            while rclpy.ok() and not self._stop_event.is_set():
                self.get_logger().info("Waiting for key to start recording...")
                self._read_single_key()

                if self._busy:
                    continue

                self._busy = True
                try:
                    self.get_logger().info("Recording... press any key to stop.")
                    audio = self._record_until_keypress()
                    wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
                    text = self._transcribe_with_elevenlabs(wav_bytes)
                    if text:
                        self._publish(text)
                    else:
                        self.get_logger().info("No speech recognised.")
                except requests.HTTPError as e:
                    self.get_logger().error(f"ElevenLabs STT HTTP error: {e}")
                except Exception as e:
                    self.get_logger().error(f"Recognition loop error: {e}")
                finally:
                    self._busy = False
        except Exception as e:
            self.get_logger().error(f"Keyboard listener stopped: {e}")

    def destroy_node(self):
        self._stop_event.set()
        super().destroy_node()


def main():
    rclpy.init()
    topic = os.environ.get("VOICE_TOPIC", "/voice/text")
    lang = os.environ.get("VOICE_LANG", "en")
    partials = _env_bool("PARTIALS", False)

    node = ElevenLabsSRPublisher(topic=topic, language=lang, publish_partials=partials)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
