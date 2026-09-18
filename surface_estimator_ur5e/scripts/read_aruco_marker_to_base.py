import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

from surface_estimator_ur5e.calibration import (
    DEFAULT_CALIBRATION,
    DEFAULT_INTRINSICS,
    load_tool_to_camera,
)
from surface_estimator_ur5e.robot_io import (
    DEFAULT_ROBOT_IP,
    connect_rtde_control,
    connect_rtde_receive,
    set_tcp_offset,
)
from surface_estimator_ur5e.transforms import make_transform, ur_pose_to_transform
from surface_estimator_ur5e.workcell_geometry import DEFAULT_TCP_OFFSET_UR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect one ArUco marker and print its pose in the robot base frame."
    )
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="UR robot IP.")
    parser.add_argument(
        "--marker-length-mm",
        type=float,
        required=True,
        help="Printed black ArUco square side length in millimeters.",
    )
    parser.add_argument(
        "--dictionary",
        default="DICT_ARUCO_ORIGINAL",
        help="OpenCV ArUco dictionary name, e.g. DICT_4X4_50 or DICT_6X6_250.",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        help="Only print this marker ID. If omitted, the first detected marker is used.",
    )
    parser.add_argument("--intrinsics", default=DEFAULT_INTRINSICS)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", help="RealSense serial number.")
    parser.add_argument(
        "--tcp-offset-ur",
        nargs=6,
        type=float,
        default=DEFAULT_TCP_OFFSET_UR,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Active UR TCP offset used during hand-eye calibration.",
    )
    parser.add_argument(
        "--no-set-tcp",
        action="store_true",
        help="Do not call RTDEControl.setTcp before reading poses.",
    )
    parser.add_argument("--once", action="store_true", help="Exit after first detection.")
    parser.add_argument("--no-gui", action="store_true", help="Do not show preview window.")
    parser.add_argument(
        "--print-period",
        type=float,
        default=0.5,
        help="Minimum seconds between printed detections.",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="",
        help=(
            "Save an averaged marker pose YAML. If no path is given, "
            "uses data/markers/aruco_<id>_in_base.yaml."
        ),
    )
    parser.add_argument(
        "--save-samples",
        type=int,
        default=50,
        help="Number of detected marker poses to average before saving.",
    )
    return parser.parse_args()


def load_intrinsics(path):
    with open(path) as file:
        data = yaml.safe_load(file)

    camera_matrix = np.array(data["camera_matrix"]["data"], dtype=float).reshape(3, 3)
    dist_coeffs = np.array(data["distortion_coefficients"]["data"], dtype=float)
    return camera_matrix, dist_coeffs


def rvec_tvec_to_transform(rvec, tvec):
    rotation, _ = cv2.Rodrigues(np.array(rvec, dtype=float).reshape(3))
    translation = np.array(tvec, dtype=float).reshape(3)
    return make_transform(rotation, translation)


def rotation_to_rpy_degrees(rotation):
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-9

    if singular:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    else:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])

    return np.degrees([roll, pitch, yaw])


def rotation_to_quaternion_xyzw(rotation):
    m = np.asarray(rotation, dtype=float)
    trace = float(np.trace(m))

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    quat = np.array([x, y, z, w], dtype=float)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm


def quaternion_xyzw_to_rotation(quat):
    x, y, z, w = np.asarray(quat, dtype=float)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def average_quaternions_xyzw(quaternions):
    quats = np.asarray(quaternions, dtype=float)
    reference = quats[0].copy()
    aligned = []
    for quat in quats:
        quat = quat.copy()
        if np.dot(quat, reference) < 0.0:
            quat *= -1.0
        aligned.append(quat)

    accumulator = np.zeros((4, 4), dtype=float)
    for quat in aligned:
        accumulator += np.outer(quat, quat)

    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    averaged = eigenvectors[:, int(np.argmax(eigenvalues))]
    if np.dot(averaged, reference) < 0.0:
        averaged *= -1.0
    return averaged / np.linalg.norm(averaged)


def float_list(values, digits=9):
    return [round(float(value), digits) for value in np.asarray(values).reshape(-1)]


def matrix_list(matrix, digits=9):
    return [[round(float(value), digits) for value in row] for row in np.asarray(matrix)]


def apply_tcp_offset(robot_ip, tcp_offset_ur):
    control = connect_rtde_control(robot_ip)
    try:
        set_tcp_offset(control, tcp_offset_ur)
        return control
    except Exception:
        try:
            control.stopScript()
        finally:
            control.disconnect()
        raise


def default_save_path(marker_id):
    name = f"aruco_{marker_id}_in_base.yaml" if marker_id is not None else "aruco_marker_in_base.yaml"
    return Path("data/markers") / name


def save_marker_pose(path, args, marker_id, marker_length_m, base_to_markers, camera_to_markers):
    path = Path(path)
    base_translations = np.array([transform[:3, 3] for transform in base_to_markers])
    base_quaternions = np.array(
        [rotation_to_quaternion_xyzw(transform[:3, :3]) for transform in base_to_markers]
    )
    camera_translations = np.array([transform[:3, 3] for transform in camera_to_markers])

    base_translation = np.median(base_translations, axis=0)
    base_quaternion = average_quaternions_xyzw(base_quaternions)
    base_rotation = quaternion_xyzw_to_rotation(base_quaternion)
    base_transform = make_transform(base_rotation, base_translation)

    payload = {
        "marker_id": int(marker_id),
        "dictionary": args.dictionary,
        "marker_length_m": round(float(marker_length_m), 9),
        "marker_origin": "center",
        "pose_in_base": {
            "translation_m": float_list(base_translation),
            "rotation_quaternion_xyzw": float_list(base_quaternion),
            "rotation_rpy_deg": float_list(rotation_to_rpy_degrees(base_rotation), digits=6),
            "rotation_matrix": matrix_list(base_rotation),
            "transform_matrix": matrix_list(base_transform),
        },
        "capture_quality": {
            "samples": int(len(base_to_markers)),
            "translation_std_mm_xyz": float_list(np.std(base_translations, axis=0) * 1000.0, digits=4),
            "translation_range_mm_xyz": float_list(
                (np.max(base_translations, axis=0) - np.min(base_translations, axis=0)) * 1000.0,
                digits=4,
            ),
        },
        "camera_observation_median": {
            "translation_m": float_list(np.median(camera_translations, axis=0)),
        },
        "source": {
            "robot_ip": args.robot_ip,
            "tcp_offset_ur": float_list(args.tcp_offset_ur),
            "set_tcp_before_capture": not args.no_set_tcp,
            "intrinsics": args.intrinsics,
            "calibration": args.calibration,
            "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def get_aruco_dictionary(dictionary_name):
    aruco = cv2.aruco
    if not hasattr(aruco, dictionary_name):
        names = sorted(name for name in dir(aruco) if name.startswith("DICT_"))
        raise ValueError(
            f"Unknown dictionary {dictionary_name}. Available examples: "
            + ", ".join(names[:8])
        )

    dictionary_id = getattr(aruco, dictionary_name)
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dictionary_id)
    return aruco.Dictionary_get(dictionary_id)


def make_detector_parameters():
    aruco = cv2.aruco
    if hasattr(aruco, "DetectorParameters_create"):
        return aruco.DetectorParameters_create()
    return aruco.DetectorParameters()


def make_aruco_detector(aruco_dict, aruco_params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    return None


def detect_markers(gray, aruco_dict, aruco_params, detector):
    if detector is not None:
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)


def estimate_single_marker_pose(corners, marker_length_m, camera_matrix, dist_coeffs):
    half = marker_length_m / 2.0
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    flags = (
        cv2.SOLVEPNP_IPPE_SQUARE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
        else cv2.SOLVEPNP_ITERATIVE
    )
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs, flags=flags
    )
    if not success and flags != cv2.SOLVEPNP_ITERATIVE:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not success:
        raise RuntimeError("cv2.solvePnP failed for detected marker.")
    return rvec.reshape(3), tvec.reshape(3)


def start_realsense(args):
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )
    pipeline.start(config)
    return pipeline


def main():
    args = parse_args()
    marker_length_m = args.marker_length_mm / 1000.0

    camera_matrix, dist_coeffs = load_intrinsics(args.intrinsics)
    tool_to_camera = load_tool_to_camera(args.calibration)
    aruco_dict = get_aruco_dictionary(args.dictionary)
    aruco_params = make_detector_parameters()
    aruco_detector = make_aruco_detector(aruco_dict, aruco_params)

    rtde_c = None
    if not args.no_set_tcp:
        rtde_c = apply_tcp_offset(args.robot_ip, args.tcp_offset_ur)
    rtde_r = connect_rtde_receive(args.robot_ip)
    pipeline = start_realsense(args)

    print("Using:")
    print(f"  dictionary: {args.dictionary}")
    print(f"  marker length: {marker_length_m:.6f} m")
    print(f"  intrinsics: {args.intrinsics}")
    print(f"  calibration: {args.calibration}")
    print(f"  tcp offset UR: {args.tcp_offset_ur}")
    print(f"  set tcp before capture: {not args.no_set_tcp}")
    print("Press q/Esc in the preview window to quit.")
    if args.save is not None:
        if args.save_samples < 1:
            raise ValueError("--save-samples must be at least 1.")
        print(f"Will save averaged marker pose after {args.save_samples} detections.")

    last_print = 0.0
    window_name = "ArUco to Robot Base"
    save_base_to_markers = []
    save_camera_to_markers = []

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers(
                gray, aruco_dict, aruco_params, aruco_detector
            )

            display = image.copy()
            detected = []

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)

                marker_ids = np.asarray(ids).reshape(-1)
                for index, marker_id_value in enumerate(marker_ids):
                    marker_id = int(marker_id_value)
                    if args.marker_id is not None and marker_id != args.marker_id:
                        continue

                    try:
                        rvec, tvec = estimate_single_marker_pose(
                            corners[index], marker_length_m, camera_matrix, dist_coeffs
                        )
                    except RuntimeError:
                        continue

                    camera_to_marker = rvec_tvec_to_transform(rvec, tvec)
                    base_to_tool = ur_pose_to_transform(rtde_r.getActualTCPPose())
                    base_to_marker = base_to_tool @ tool_to_camera @ camera_to_marker

                    base_xyz = base_to_marker[:3, 3]
                    base_rpy = rotation_to_rpy_degrees(base_to_marker[:3, :3])
                    camera_xyz = camera_to_marker[:3, 3]
                    detected.append((marker_id, base_to_marker, camera_to_marker, base_xyz, base_rpy, camera_xyz))

                    cv2.drawFrameAxes(
                        display,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        marker_length_m * 0.5,
                    )

                    if args.marker_id is None:
                        break

            if args.save is not None and detected:
                marker_id, base_to_marker, camera_to_marker, *_unused = detected[0]
                save_base_to_markers.append(base_to_marker.copy())
                save_camera_to_markers.append(camera_to_marker.copy())
                sample_count = len(save_base_to_markers)
                if sample_count == 1 or sample_count % 10 == 0 or sample_count == args.save_samples:
                    print(
                        f"Collected marker samples for save: {sample_count}/{args.save_samples}",
                        flush=True,
                    )
                if sample_count >= args.save_samples:
                    save_path = Path(args.save) if args.save else default_save_path(marker_id)
                    save_marker_pose(
                        save_path,
                        args,
                        marker_id,
                        marker_length_m,
                        save_base_to_markers,
                        save_camera_to_markers,
                    )
                    print(f"Saved marker pose to {save_path}", flush=True)
                    break

            now = time.monotonic()
            if detected and now - last_print >= args.print_period:
                for marker_id, _base_to_marker, _camera_to_marker, base_xyz, base_rpy, camera_xyz in detected:
                    print(
                        f"marker_id={marker_id} "
                        f"base_xyz_m=[{base_xyz[0]: .6f}, {base_xyz[1]: .6f}, {base_xyz[2]: .6f}] "
                        f"base_rpy_deg=[{base_rpy[0]: .3f}, {base_rpy[1]: .3f}, {base_rpy[2]: .3f}] "
                        f"camera_xyz_m=[{camera_xyz[0]: .6f}, {camera_xyz[1]: .6f}, {camera_xyz[2]: .6f}]",
                        flush=True,
                    )
                last_print = now
                if args.once and args.save is None:
                    break

            if not args.no_gui:
                if detected:
                    marker_id, _base_to_marker, _camera_to_marker, base_xyz, _base_rpy, _camera_xyz = detected[0]
                    text = (
                        f"id {marker_id} base xyz m: "
                        f"{base_xyz[0]:.3f}, {base_xyz[1]:.3f}, {base_xyz[2]:.3f}"
                    )
                else:
                    text = "No target marker detected"

                cv2.rectangle(display, (20, 20), (900, 75), (0, 0, 0), -1)
                cv2.putText(
                    display,
                    text,
                    (35, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if rtde_r is not None and hasattr(rtde_r, "disconnect"):
            rtde_r.disconnect()
        if rtde_c is not None and hasattr(rtde_c, "stopScript"):
            rtde_c.stopScript()
        if rtde_c is not None and hasattr(rtde_c, "disconnect"):
            rtde_c.disconnect()


if __name__ == "__main__":
    main()
