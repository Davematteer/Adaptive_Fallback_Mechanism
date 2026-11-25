"""TCP server the host connects to. Receives frames, returns driving commands.

The server knows nothing about driving: it hands each frame to `on_frame` and
sends back whatever command that returns.

Wire format (must match host/communication/jetson_connection.py):
    [4-byte big-endian header length][JSON header][payload of header["size"] bytes]
"""

import json
import socket
import struct
import threading
import time


def send_msg(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    header = dict(header, size=len(payload))
    raw = json.dumps(header).encode()
    sock.sendall(struct.pack("!I", len(raw)) + raw + payload)


def recv_msg(sock: socket.socket) -> tuple[dict, bytes]:
    (length,) = struct.unpack("!I", _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, length))
    return header, _recv_exact(sock, header.get("size", 0))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed by peer")
        buf.extend(chunk)
    return bytes(buf)


class CommunicationServer:
    """Two threads per host connection, so a failing pipeline can't take communication down with it:

    - receiver (this thread): reads messages, answers "hello", passes on fault requests,
      and keeps only the NEWEST frame for the pipeline.
    - pipeline worker: takes the newest frame, runs on_frame, sends the command back.

    If the pipeline hangs, only the worker is stuck: the receiver still notices when the host
    disconnects (Stop), calls on_disconnect to clear the fault, and the app is healthy again
    for the next experiment, with no restart needed.
    """

    def __init__(self, on_frame, on_fault, on_disconnect, host: str = "0.0.0.0", port: int = 5555):
        self.on_frame = on_frame            # (header, payload) -> command dict
        self.on_fault = on_fault            # (fault name) -> None
        self.on_disconnect = on_disconnect  # () -> None, clears any injected fault
        self.host, self.port = host, port
        self.muted = False                  # True = keep running but stop replying (comm_failure fault)

    def serve_forever(self) -> None:
        """Accept one host at a time; when it disconnects, wait for the next."""
        with socket.create_server((self.host, self.port)) as server:
            log(f"listening on {self.host}:{self.port}")
            while True:
                conn, addr = server.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                log(f"host connected from {addr[0]}")
                self.muted = False  # a new experiment starts healthy
                self._handle(conn)
                conn.close()
                self.on_disconnect()
                log("host disconnected; faults cleared, ready for the next experiment")

    def _handle(self, conn: socket.socket) -> None:
        # Per-connection state, so a worker left over from an earlier session can't touch this one.
        session = Session(conn)
        threading.Thread(target=self._pipeline_worker, args=(session,), daemon=True).start()
        try:
            while True:
                header, payload = recv_msg(conn)
                if header["type"] == "frame":
                    with session.wake:
                        session.latest = (header, payload)  # replaces any older unprocessed frame
                        session.wake.notify()
                elif header["type"] == "hello":  # host checking that this app is alive
                    session.send({"type": "ready"})
                elif header["type"] == "fault":
                    self.on_fault(header["fault"])
        except (ConnectionError, OSError, ValueError):
            pass  # host closed the connection (or sent garbage): end this session
        finally:
            with session.wake:
                session.closed = True
                session.wake.notify()

    def _pipeline_worker(self, session: "Session") -> None:
        while True:
            with session.wake:
                while session.latest is None and not session.closed:
                    session.wake.wait()
                if session.closed:
                    return
                header, payload = session.latest
                session.latest = None
            command = self.on_frame(header, payload)  # may hang if a fault was injected
            if session.closed:
                return
            if not self.muted:
                try:
                    session.send(dict(command, type="command", frame_id=header["frame_id"]))
                except OSError:
                    return


class Session:
    """State for one host connection."""

    def __init__(self, conn: socket.socket):
        self.conn = conn
        self.latest = None   # newest unprocessed (header, payload); older ones are dropped
        self.closed = False
        self.wake = threading.Condition()
        self._send_lock = threading.Lock()  # receiver and worker both send

    def send(self, header: dict) -> None:
        with self._send_lock:
            send_msg(self.conn, header)


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {text}", flush=True)
