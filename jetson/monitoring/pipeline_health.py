"""Heartbeat for the driving pipeline, plus the injectable faults that break it.

The heartbeat only advances after a full pipeline step (perception -> planning
-> control) has completed. If the pipeline hangs or crashes, the heartbeat stops,
and that is what the FPGA will watch.
"""

import os
import time


class PipelineHealth:
    def __init__(self):
        self.heartbeat = 0
        self.last_beat_at = None
        self.pending_fault = None

    def beat(self) -> int:
        """Call ONLY after a complete, successful pipeline step."""
        self.heartbeat += 1
        self.last_beat_at = time.monotonic()
        # Later: send self.heartbeat to the FPGA over UART here.
        return self.heartbeat

    def inject(self, fault: str) -> None:
        """Queue a fault; it takes effect at the start of the next pipeline step."""
        print(f"{time.strftime('%H:%M:%S')}  fault requested: {fault}", flush=True)
        self.pending_fault = fault

    def apply_pending_fault(self) -> None:
        """Called at the start of every pipeline step to simulate the requested failure."""
        while self.pending_fault == "pipeline_hang":  # stuck, like a hung GPU/inference call:
            time.sleep(0.05)                          # no more heartbeats until clear() is called
        if self.pending_fault == "process_crash":
            os._exit(1)  # die immediately, without cleanup, like a real crash

    def clear(self) -> None:
        """End any injected fault (called when the host disconnects at the end of an experiment)."""
        self.pending_fault = None
