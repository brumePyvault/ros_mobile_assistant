#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.


import azure.cognitiveservices.speech as speechsdk

def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "")
    if not v:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

class AzureSRPublisher(Node):
    def __init__(self,
                 topic: str = "/voice/text",
                 language: str = "en-GB",
                 device_name: str | None = None,
                 publish_partials: bool = True):
        super().__init__("azure_sr_to_topic")
        self.pub = self.create_publisher(String, topic, 10)

        # --- Azure config ---
        key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
        region = os.environ.get("AZURE_SPEECH_REGION", "").strip()
        if not key or not region:
            raise RuntimeError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION in environment.")

        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        speech_config.set_property(
    speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "2500"
)
        speech_config.speech_recognition_language = language
        # Optional: better punctuation/formatting
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption, "TrueText"
        )

        if device_name and device_name.lower() != "default":
            audio_config = speechsdk.audio.AudioConfig(device_name=device_name)
        else:
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

        self.recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        self.publish_partials = publish_partials

        self.get_logger().info(f"Azure Speech: region={region} | lang={language}")
        self.get_logger().info(f"Mic device: {device_name if device_name else 'default'}")
        self.get_logger().info(f"Publishing recognised text to: {topic}")

        # --- Wire up events ---
        # Final results
        self.recognizer.recognized.connect(self._on_recognized)
        # Intermediate hypotheses
        if self.publish_partials:
            self.recognizer.recognizing.connect(self._on_recognizing)
        # Errors / cancellations
        self.recognizer.canceled.connect(self._on_canceled)
        self.recognizer.session_started.connect(lambda evt: self.get_logger().info("Session started."))
        self.recognizer.session_stopped.connect(lambda evt: self.get_logger().info("Session stopped."))

        # Start continuous recognition (non-blocking)
        self.recognizer.start_continuous_recognition_async()
        self.get_logger().info("Listening… speak a phrase and pause.")

    def _publish(self, text: str, tag: str = ""):
        text = text.strip()
        if not text:
            return
        msg = String()
        msg.data = text if not tag else f"[{tag}] {text}"
        self.pub.publish(msg)
        self.get_logger().info(f"→ {msg.data}")

    def _on_recognizing(self, evt: speechsdk.SessionEventArgs):
        # evt.result.text contains partial transcription
        try:
            # self._publish(evt.result.text, tag="partial")
            pass
        except Exception as e:
            self.get_logger().error(f"Recognizing handler error: {e}")

    def _on_recognized(self, evt: speechsdk.SessionEventArgs):
        try:
            result = evt.result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                self._publish(result.text)
            elif result.reason == speechsdk.ResultReason.NoMatch:
                # No audible speech or could not match
                pass
        except Exception as e:
            self.get_logger().error(f"Recognized handler error: {e}")

    def _on_canceled(self, evt: speechsdk.SessionEventArgs):
        details = evt.result.cancellation_details
        self.get_logger().error(f"Recognition canceled: reason={details.reason} | error={details.error_details}")

    def shutdown(self):
        try:
            self.recognizer.stop_continuous_recognition_async().get()
        except Exception:
            pass

def main():
    rclpy.init()
    topic = os.environ.get("VOICE_TOPIC", "/voice/text")
    lang = "en-GB"
    mic_name = os.environ.get("MIC_DEVICE_NAME", "").strip()
    partials = _env_bool("PARTIALS", True)

    node = AzureSRPublisher(topic=topic, language=lang,
                            device_name=mic_name if mic_name else None,
                            publish_partials=partials)
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
