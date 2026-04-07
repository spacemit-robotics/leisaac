"""Network-based SO101 Leader device.

Connects to a remote leader_sender.py TCP server running on the local PC,
receives normalized joint positions over the network, and provides them
to the Isaac Sim teleoperation loop — as a drop-in replacement for SO101Leader
that does NOT require a local serial port.
"""

import json
import socket
import threading
import time

from ..device_base import Device


class NetworkSO101Leader(Device):
    """A network-based SO101 Leader device for teleoperation.

    Instead of reading motor positions from a local USB serial port, this device
    connects to a TCP server (leader_sender.py) on the local PC that streams
    the leader arm's joint states over the network.
    """

    def __init__(
        self,
        env,
        host: str = "10.0.91.83",
        port: int = 5050,
    ):
        """
        Args:
            env: The Isaac Sim environment.
            host: IP address of the local PC running leader_sender.py.
            port: TCP port of the leader_sender.py server.
        """
        # Set attributes BEFORE super().__init__() because it calls _add_device_control_description()
        self._host = host
        self._port = port

        # latest joint state received from the network (thread-safe)
        self._joint_state: dict[str, float] | None = None
        self._motor_limits: dict[str, tuple[float, float]] | None = None
        self._lock = threading.Lock()

        # networking
        self._sock: socket.socket | None = None
        self._recv_thread: threading.Thread | None = None
        self._running = False

        super().__init__(env, "so101_leader")

        # connect
        self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self):
        """Connect to the remote leader_sender.py TCP server."""
        print(f"[NetworkSO101Leader] Connecting to {self._host}:{self._port} ...")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.connect((self._host, self._port))
        self._running = True

        # Read the first message – should be the config (motor_limits)
        buf = b""
        while b"\n" not in buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("Connection closed before receiving config")
            buf += chunk
        line, remainder = buf.split(b"\n", 1)
        config = json.loads(line)
        if config.get("type") == "config":
            self._motor_limits = {k: tuple(v) for k, v in config["motor_limits"].items()}
            print(f"[NetworkSO101Leader] Received motor limits: {list(self._motor_limits.keys())}")
        else:
            raise ValueError(f"Expected config message, got: {config.get('type')}")

        # Start background receiver thread
        self._recv_thread = threading.Thread(target=self._recv_loop, args=(remainder,), daemon=True)
        self._recv_thread.start()
        print(f"[NetworkSO101Leader] Connected and receiving joint states.")

    def _recv_loop(self, initial_buf: bytes = b""):
        """Background thread: continuously read JSON lines from the socket."""
        buf = initial_buf
        while self._running:
            try:
                chunk = self._sock.recv(4096)
                if not chunk:
                    print("[NetworkSO101Leader] Server closed connection.")
                    break
                buf += chunk
                # Process all complete lines
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("type") == "state":
                        with self._lock:
                            self._joint_state = msg["joint_state"]
            except OSError:
                break
        self._running = False

    def disconnect(self):
        """Disconnect from the TCP server."""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=2.0)
        print("[NetworkSO101Leader] Disconnected.")

    # ------------------------------------------------------------------
    # Device interface (same contract as SO101Leader)
    # ------------------------------------------------------------------

    def get_device_state(self) -> dict[str, float]:
        """Return the latest joint state received from the network.

        Returns:
            dict mapping motor name -> normalized position, e.g.
            {"shoulder_pan": 12.3, "shoulder_lift": -5.1, ...}
        """
        with self._lock:
            if self._joint_state is None:
                # Return zeros if we haven't received a state yet
                return {
                    "shoulder_pan": 0.0,
                    "shoulder_lift": 0.0,
                    "elbow_flex": 0.0,
                    "wrist_flex": 0.0,
                    "wrist_roll": 0.0,
                    "gripper": 0.0,
                }
            return dict(self._joint_state)

    def input2action(self):
        """Build the action dict expected by preprocess_device_action().

        Sets ``so101_leader: True`` so the dispatcher routes to
        ``convert_action_from_so101_leader()``.
        """
        ac_dict = super().input2action()
        ac_dict["motor_limits"] = self._motor_limits
        return ac_dict

    @property
    def motor_limits(self) -> dict[str, tuple[float, float]]:
        return self._motor_limits

    def _add_device_control_description(self):
        self._display_controls_table.add_row([
            "network-so101-leader",
            f"receiving joint states from {self._host}:{self._port}",
        ])
        self._display_controls_table.add_row([
            "[TIPS]",
            "Make sure leader_sender.py is running on the local PC before starting.",
        ])

    def __del__(self):
        """Clean up network resources."""
        self.disconnect()
        super().__del__()
