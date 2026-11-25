"""Asks the Jetson to deliberately fail in a chosen way."""

import time

# Fault name -> what the Jetson does when it receives it (see jetson/main.py).
FAULTS = {
    "pipeline_hang": "Driving pipeline stops making progress (simulates a GPU/inference hang).",
    "process_crash": "Driving process exits immediately.",
    "comm_failure": "Jetson keeps running but stops sending commands.",
}


class FaultInjector:
    def __init__(self, connection):
        self.connection = connection

    def inject(self, fault: str) -> float:
        """Send the fault request and return the wall-clock time it was sent (for logging)."""
        if fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}; choose from {sorted(FAULTS)}")
        sent_at = time.time()
        self.connection.send_control({"type": "fault", "fault": fault})
        return sent_at
