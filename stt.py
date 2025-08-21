import os, json, time, queue, threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sounddevice as sd
import vosk

class MicToTopic(Node):
    def __init__(self, topic="/voice/text", rate=16000, device=None):
        super().__init__('mic_to_text')
        self.pub = self.create_publisher(String, topic, 10)
        self.get_logger().info(f"Publishing to {topic}")

        model_dir = os.environ.get("VOSK_MODEL")
        if not model_dir or not os.path.isdir(model_dir):
            self.get_logger().error("Set VOSK_MODEL to your model folder"); raise SystemExit(2)
        self.rec = vosk.KaldiRecognizer(vosk.Model(model_dir), rate)

        self.rate = rate
        self.device = device
        self.q = queue.Queue()

        # one thread grabs audio, one processes it
        threading.Thread(target=self._mic_loop, daemon=True).start()
        threading.Thread(target=self._process_loop, daemon=True).start()

    def _mic_loop(self):
        def cb(indata, frames, t, status):
            if status: pass
            self.q.put(bytes(indata))
        kwargs = dict(samplerate=self.rate, blocksize=self.rate//2,
                      dtype='int16', channels=1, callback=cb)
        if self.device is not None:
            kwargs["device"] = self.device
        with sd.RawInputStream(**kwargs):
            self.get_logger().info("Mic open. Speak a phrase and pause.")
            while rclpy.ok():
                time.sleep(0.01)

    def _process_loop(self):
        while rclpy.ok():
            try:
                data = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.rec.AcceptWaveform(data):
                text = (json.loads(self.rec.Result()).get("text") or "").strip()
                if text:
                    msg = String(); msg.data = text
                    self.pub.publish(msg)
                    self.get_logger().info("→ " + text)

def main():
    rclpy.init()
    node = MicToTopic(topic="/voice/text", rate=16000, device=None)  # set device index if needed
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
