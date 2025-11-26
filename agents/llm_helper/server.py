# run_simulation.py or utils/lmdeploy_server.py
import subprocess
import time
import requests
from threading import Thread
import atexit
import signal
import os

class LMDeployServer:
    def __init__(
        self,
        model_path: str,
        port: int = 23333,
        tp: int = 1,
        backend: str = "turbomind",
        quant_policy: str = None,  # e.g., "w4a16"
        max_batch_size: int = 128,
        host: str = "127.0.0.1"
    ):
        self.model_path = model_path
        self.port = port
        self.host = host
        self.url = f"http://{host}:{port}/v1"
        self.process = None

        cmd = [
            "lmdeploy", "serve", "api",
            "--model", model_path,
            "--backend", backend,
            "--tp", str(tp),
            "--host", host,
            "--port", str(port),
            "--max-batch-size", str(max_batch_size),
        ]
        if quant_policy:
            cmd += ["--quant-policy", quant_policy]

        self.cmd = cmd

    def start(self):
        print(f"[LMDeploy] Starting server: {' '.join(self.cmd)}")
        self.process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # Allows killing process group
        )

        # Health check loop
        def wait_for_server():
            for _ in range(60):  # 60 seconds timeout
                try:
                    resp = requests.get(f"{self.url}/models")
                    if resp.status_code == 200:
                        print(f"[LMDeploy] Server ready at {self.url}")
                        return
                except:
                    pass
                time.sleep(1)
            print("[LMDeploy] Server failed to start!")
            if self.process:
                self.process.terminate()

        Thread(target=wait_for_server, daemon=True).start()

    def stop(self):
        if self.process:
            print("[LMDeploy] Shutting down server...")
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=10)
            print("[LMDeploy] Server stopped.")

    def is_ready(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/models", timeout=3)
            return resp.status_code == 200
        except:
            return False