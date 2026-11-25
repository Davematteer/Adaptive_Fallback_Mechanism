"""Decides who is allowed to drive the vehicle: the Jetson or the fallback.

Rule: once fallback is latched, Jetson commands are ignored until an explicit
reset, even if the Jetson recovers and starts sending commands again.

TEMPORARY: until the FPGA fallback path exists, this class also acts as a simple
watchdog (stale Jetson commands -> latch fallback). When the FPGA is wired in,
call latch_fallback() from the FPGA signal and set_fallback_command() with its
commands instead.
"""

import time
from enum import Enum


class Owner(str, Enum):
    JETSON = "jetson"
    FALLBACK = "fallback"


SAFE_STOP = {"steer": 0.0, "throttle": 0.0, "brake": 1.0, "reverse": False}  # the fallback never reverses


class CommandArbitrator:
    def __init__(self, stale_after_s: float = 0.5):
        self.stale_after_s = stale_after_s
        # False = baseline experiment: failures are detected and logged, but nothing takes
        # over, so the vehicle keeps executing the Jetson's last command (and likely crashes).
        self.fallback_enabled = True
        self.reset()

    def reset(self) -> None:
        self.owner = Owner.JETSON
        self.fallback_command = dict(SAFE_STOP)
        self.latch_reason = None
        self.latched_at = None
        self.failure_detected_at = None  # first time the Jetson was seen to stop answering

    def latch_fallback(self, reason: str) -> None:
        if self.owner is Owner.FALLBACK:
            return  # already latched; keep the first reason and time
        self.owner = Owner.FALLBACK
        self.latch_reason = reason
        self.latched_at = time.time()

    def set_fallback_command(self, command: dict) -> None:
        """For later: commands produced by the FPGA fallback controller."""
        self.fallback_command = command

    def select(self, jetson_command: dict | None, unanswered_s: float | None) -> dict:
        """Return the command that should be applied to the vehicle this step."""
        # Watchdog stand-in: only armed once the Jetson has sent at least one command,
        # so the start-up period doesn't count as a failure.
        # unanswered_s = how long the Jetson has been holding a frame without replying.
        if self.owner is Owner.JETSON and unanswered_s is not None and unanswered_s > self.stale_after_s:
            reason = f"Jetson did not answer a frame for {unanswered_s:.2f} s"
            if self.failure_detected_at is None:
                self.failure_detected_at = time.time()
                self.latch_reason = reason
            if self.fallback_enabled:
                self.latch_fallback(reason)

        if self.owner is Owner.FALLBACK:
            return self.fallback_command
        # Healthy Jetson, or fallback disabled: apply the Jetson's latest command. If the
        # Jetson has failed, that is its LAST command, held, like real actuators would.
        return jetson_command or SAFE_STOP  # nothing received yet: hold still
