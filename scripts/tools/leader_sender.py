#!/usr/bin/env python3
"""
Local SO101 Leader Arm Network Sender

Reads the SO101 Leader arm via USB serial and streams normalized joint positions
over a TCP socket to the remote Isaac Sim server.

Usage (on local PC):
    python scripts/tools/leader_sender.py --port /dev/ttyACM0 --listen-port 5050
    python scripts/tools/leader_sender.py --port /dev/ttyACM0 --listen-port 5050 --recalibrate

Protocol:
    - TCP server listens on --listen-port
    - Sends JSON lines: {"joint_state": {...}, "motor_limits": {...}, "started": bool, "reset": bool, "success": bool}
    - Remote client connects and receives joint states in real-time
"""

import argparse
import json
import os
import socket
import sys
import threading
import time

# Add project root to path — import motor modules directly to avoid
# triggering leisaac.__init__ which depends on isaaclab_tasks.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "source", "leisaac", "leisaac", "devices", "lerobot"))

from common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from common.motors import (
    FeetechMotorsBus,
    Motor,
    MotorCalibration,
    MotorNormMode,
    OperatingMode,
)

# Motor limits — same as leisaac.assets.robots.lerobot.SO101_FOLLOWER_MOTOR_LIMITS
# Duplicated here so leader_sender.py can run on a local PC without Isaac Sim.
SO101_FOLLOWER_MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}


class LeaderArmReader:
    """Reads SO101 Leader arm motor positions via USB serial."""

    def __init__(self, port: str, recalibrate: bool = False, calibration_file_name: str = "so101_leader.json"):
        self.port = port
        self.calibration_path = os.path.join(
            PROJECT_ROOT, "source", "leisaac", "leisaac", "devices", "lerobot", ".cache", calibration_file_name
        )
        self._motor_limits = SO101_FOLLOWER_MOTOR_LIMITS

        if not os.path.exists(self.calibration_path) or recalibrate:
            self._calibrate()

        calibration = self._load_calibration()
        self._bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration,
        )

    def connect(self):
        self._bus.connect()
        self._bus.disable_torque()
        self._bus.configure_motors()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        print(f"[LEADER] Connected to {self.port}")

    def disconnect(self):
        self._bus.disconnect()
        print("[LEADER] Disconnected")

    def read_positions(self) -> dict[str, float]:
        """Read normalized motor positions."""
        return self._bus.sync_read("Present_Position")

    @property
    def motor_limits(self) -> dict[str, tuple[float, float]]:
        return self._motor_limits

    def _calibrate(self):
        print("\n[LEADER] Running calibration...")
        bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
        )
        bus.connect()
        bus.disable_torque()
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input("Move SO101-Leader to the middle of its range of motion and press ENTER...")
        homing_offset = bus.set_half_turn_homings()
        print("Move all joints sequentially through their entire ranges of motion.")
        print("Recording positions. Press ENTER to stop...")
        range_mins, range_maxes = bus.record_ranges_of_motion()

        calibration = {}
        for motor_name, m in bus.motors.items():
            calibration[motor_name] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offset[motor_name],
                range_min=range_mins[motor_name],
                range_max=range_maxes[motor_name],
            )
        bus.write_calibration(calibration)
        self._save_calibration(calibration)
        print(f"[LEADER] Calibration saved to {self.calibration_path}")
        bus.disconnect()

    def _load_calibration(self) -> dict[str, MotorCalibration]:
        with open(self.calibration_path) as f:
            json_data = json.load(f)
        calibration = {}
        for motor_name, motor_data in json_data.items():
            calibration[motor_name] = MotorCalibration(
                id=int(motor_data["id"]),
                drive_mode=int(motor_data["drive_mode"]),
                homing_offset=int(motor_data["homing_offset"]),
                range_min=int(motor_data["range_min"]),
                range_max=int(motor_data["range_max"]),
            )
        return calibration

    def _save_calibration(self, calibration: dict[str, MotorCalibration]):
        save_data = {
            k: {
                "id": v.id,
                "drive_mode": v.drive_mode,
                "homing_offset": v.homing_offset,
                "range_min": v.range_min,
                "range_max": v.range_max,
            }
            for k, v in calibration.items()
        }
        os.makedirs(os.path.dirname(self.calibration_path), exist_ok=True)
        with open(self.calibration_path, "w") as f:
            json.dump(save_data, f, indent=4)


def serve_leader(reader: LeaderArmReader, listen_port: int, rate_hz: float = 60.0):
    """TCP server that streams joint states to connected clients."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", listen_port))
    server.listen(1)
    print(f"[SERVER] Listening on 0.0.0.0:{listen_port}")
    print(f"[SERVER] Streaming at {rate_hz} Hz")
    print(f"[SERVER] Waiting for remote Isaac Sim to connect...")

    interval = 1.0 / rate_hz

    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[SERVER] Client connected from {addr}")

        # Send motor limits once on connection
        limits_msg = {
            "type": "config",
            "motor_limits": {k: list(v) for k, v in reader.motor_limits.items()},
        }
        try:
            conn.sendall((json.dumps(limits_msg) + "\n").encode())
        except (BrokenPipeError, ConnectionResetError):
            print(f"[SERVER] Client {addr} disconnected during config")
            conn.close()
            continue

        print(f"[SERVER] Streaming joint states... (press Ctrl+C to stop)")
        try:
            while True:
                t0 = time.monotonic()
                positions = reader.read_positions()

                msg = {
                    "type": "state",
                    "joint_state": {k: float(v) for k, v in positions.items()},
                    "timestamp": time.time(),
                }
                conn.sendall((json.dumps(msg) + "\n").encode())

                elapsed = time.monotonic() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except (BrokenPipeError, ConnectionResetError, OSError):
            print(f"[SERVER] Client {addr} disconnected")
            conn.close()
        except KeyboardInterrupt:
            print("\n[SERVER] Stopped by user")
            conn.close()
            break

    server.close()


def main():
    parser = argparse.ArgumentParser(description="SO101 Leader Arm Network Sender")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port for Leader arm")
    parser.add_argument("--listen-port", type=int, default=5050, help="TCP port to listen on (default: 5050)")
    parser.add_argument("--rate", type=float, default=60.0, help="Streaming rate in Hz (default: 60)")
    parser.add_argument("--recalibrate", action="store_true", help="Force recalibration")
    args = parser.parse_args()

    reader = LeaderArmReader(port=args.port, recalibrate=args.recalibrate)
    reader.connect()

    try:
        serve_leader(reader, listen_port=args.listen_port, rate_hz=args.rate)
    finally:
        reader.disconnect()


if __name__ == "__main__":
    main()
