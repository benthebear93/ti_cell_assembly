# Surface Estimator UR5e

Small offline Python package for estimating a planar surface from manually measured UR5e contact poses and visualizing the result with `viser`.

The intended workflow is:

1. Manually move a ceiling-mounted UR5e with a Robotiq Hand-E gripper until the TCP or tool tip contacts a flat surface.
2. Record at least three robot joint poses and Cartesian TCP poses.
3. Estimate the contacted surface plane from the measured contact positions.
4. Inspect the fit, contact points, plane normal, TCP frames, and a simple UR5e visualization in a browser.

This repository does not require ROS or a real robot connection.

## Install Without uv

From this repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If you prefer installing only from `requirements.txt` for local module execution:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Install With uv

Install `uv`, then from this repository root:

```bash
uv sync
```

## Contact YAML Format

The main input format is YAML. See [data/example_contacts.yaml](data/example_contacts.yaml).

The bundled example assumes the UR5e base frame is mounted on the ceiling and its
positive Z direction points down toward the table. The table contact points are
therefore around `z=+1.0m` in the robot base frame, while the visualizer displays
that table below the ceiling mount.

Each contact contains:

- `q_rad`: UR5e joint pose in radians, ordered according to `robot.joint_order`
- `tcp_position_m`: TCP position in the robot base frame
- `tcp_orientation_xyzw`: TCP orientation quaternion in `xyzw` order
- optional metadata such as `name` and `note`

## TCP Pose vs Contact Point

The recorded Cartesian pose may be the robot TCP, not the true physical point touching the surface. The package computes:

```python
contact_point = tcp_position + R_tcp @ contact_offset
```

where `contact_offset` is configured in YAML as `tool.contact_offset_m` and is expressed in the TCP frame. If your recorded Cartesian pose already corresponds to the physical contact point, set the offset to `[0.0, 0.0, 0.0]`.

## Plane Estimation

With exactly three contact points, the plane normal is estimated from the cross product of two edges. With more than three points, a best-fit plane is estimated using PCA/SVD. The output includes:

- centroid
- unit normal
- plane equation `n^T x + d = 0`
- per-point signed distances
- RMS fitting error

The normal direction can be controlled by `normal_direction`, `normal_direction_vector`, and `flip_normal` in YAML or with CLI flags. This is useful for a ceiling-mounted robot, where the floor normal sign depends on the robot base convention.

## Validate Input

Without uv:

```bash
surface-estimator validate --input data/example_contacts.yaml
```

or:

```bash
PYTHONPATH=src python -m surface_estimator_ur5e.cli validate --input data/example_contacts.yaml
```

With uv:

```bash
uv run surface-estimator validate --input data/example_contacts.yaml
```

This checks that the YAML is well formed, contains at least three contacts, has six joint values per contact, has valid quaternions, and that the contact points are not collinear.

## Run Estimation

Without uv:

```bash
surface-estimator estimate --input data/example_contacts.yaml
```

For your measured file from the parent directory:

```bash
surface-estimator estimate --input ../ur5e_hande_3_surface_contacts.yaml
```

With uv:

```bash
uv run surface-estimator estimate --input data/example_contacts.yaml
```

## Run Visualization

Without uv:

```bash
surface-estimator visualize --input data/example_contacts.yaml
```

For your measured file from the parent directory:

```bash
surface-estimator visualize --input ../ur5e_hande_3_surface_contacts.yaml
```

With uv:

```bash
uv run surface-estimator visualize --input data/example_contacts.yaml
```

or:

```bash
uv run python scripts/visualize_example.py
```

Without uv, the script command is:

```bash
python scripts/visualize_example.py
```

Open the printed `viser` URL in your browser. The scene shows contact points, the estimated plane, its normal, robot base/TCP/surface frames, and a simple fallback UR5e kinematic sketch.

The visualizer first tries to load a real UR5e mesh through `robot_descriptions` and `yourdfpy`, then falls back to the simple sketch if the mesh package cannot be downloaded or loaded. The first mesh run may need internet access because `robot_descriptions` fetches and caches the UR5e description.

For ceiling-mounted data, the visualization applies a display-only rotation so the robot base appears on a ceiling plane at `z=0` and the contacted surface appears below it. Plane estimation and printed numeric results remain in the original robot base frame.

## Capture From A Real UR5e

The package includes a read-only RTDE capture path for a real UR5e. It uses `ur_rtde` receive only, so it does not send motion commands to the robot.

Default robot IP:

```text
192.168.0.24
```

Check the live robot state:

```bash
uv run surface-estimator read-robot --robot-ip 192.168.0.24
```

Manually move the robot so the gripper/TCP touches the surface, then capture the pose:

```bash
uv run surface-estimator capture-contact \
  --robot-ip 192.168.0.24 \
  --output ../ur5e_hande_live_contacts.yaml \
  --name contact_1 \
  --note "manual perpendicular contact"
```

Repeat for at least three contacts:

```bash
uv run surface-estimator capture-contact --output ../ur5e_hande_live_contacts.yaml --name contact_2
uv run surface-estimator capture-contact --output ../ur5e_hande_live_contacts.yaml --name contact_3
```

Then estimate and visualize:

```bash
uv run surface-estimator validate --input ../ur5e_hande_live_contacts.yaml
uv run surface-estimator estimate --input ../ur5e_hande_live_contacts.yaml
uv run surface-estimator visualize --input ../ur5e_hande_live_contacts.yaml
```

The `capture-contact` command stores `q_rad`, `q_deg`, TCP position, UR rotation vector, and quaternion. It creates the YAML file if needed and appends each new contact.

## Run A Slow Joint Sequence

The repository includes your joint sequence in [data/ur5e_motion_sequence.yaml](data/ur5e_motion_sequence.yaml).

Preview the motion offline in `viser` before moving the real robot:

```bash
uv run surface-estimator preview-sequence --input data/ur5e_motion_sequence.yaml
```

or:

```bash
uv run python scripts/preview_motion_sequence.py
```

This animates the UR5e mesh through interpolated joint-space waypoints and draws the `tool0` path and waypoint markers. It does not connect to the robot.

Then do a dry run of the command that would move the robot:

```bash
uv run surface-estimator run-sequence --input data/ur5e_motion_sequence.yaml --robot-ip 192.168.0.24
```

or:

```bash
uv run python scripts/run_motion_sequence.py
```

If the printed plan is correct and the robot workspace is clear, execute slowly:

```bash
uv run surface-estimator run-sequence \
  --input data/ur5e_motion_sequence.yaml \
  --robot-ip 192.168.0.24 \
  --speed-rad-s 0.05 \
  --acceleration-rad-s2 0.05 \
  --dwell-s 0.5 \
  --execute
```

The script equivalent is:

```bash
uv run python scripts/run_motion_sequence.py --execute
```

This command sends `moveJ` commands through `RTDEControlInterface`. Keep the teach pendant/emergency stop reachable, use reduced mode if appropriate, and confirm the path is collision-free before adding `--execute`.

## Export A MuJoCo Scene

Generate a fixed MJCF scene with the ceiling-mounted UR5e, an attached Hand-E
gripper, contact markers, a TI tray, and a table located at the estimated
contact height:

```bash
uv run surface-estimator export-mujoco \
  --input data/example_contacts.yaml \
  --output models/ur5e_hande_table_scene.xml
```

The MuJoCo world uses the same display convention as the browser visualizer:
the robot base is fixed at the ceiling origin `z=0`, and the fitted table is
below it. The MuJoCo table is flattened to the robot-base plane so the ceiling
mount and table are parallel; the raw fitted normal is still recorded in the
generated XML comment. The generated scene is static; it is intended as a
geometry/context model, not a collision-accurate actuated robot model. If the optional `mujoco`
Python package is installed, add `--compile-check` to compile-check the MJCF
before writing it.

## Run The MuJoCo Grasp/Insert Demo

Generate the actuated grasp scene, then run the scripted grasp and insert motion:

```bash
uv run surface-estimator export-mujoco-grasp \
  --input data/example_contacts.yaml \
  --output models/ur5e_hande_tray_grasp_scene.xml \
  --compile-check

uv run surface-estimator run-mujoco-grasp \
  --mjcf models/ur5e_hande_tray_grasp_scene.xml
```

The generated robot mounts `assets/realsense_d435i` on the wrist/tool frame.
`run-mujoco-grasp` keeps the main MuJoCo viewer in the overview-style free
camera by default, and opens separate OpenCV `RealSense RGB` and `RealSense
Depth` windows plus an Open3D `RealSense Point Cloud` window from the
wrist-mounted `realsense_rgb` camera. Use `--no-realsense-windows` to disable
those extra windows, or `--viewer-camera overview` to force the main MuJoCo
viewer to the fixed overview camera.

## UR5e + Robotiq Hand-E + RealSense Assets

The MuJoCo export uses local UR5e MJCF meshes in `assets/universal_robots_ur5e`,
local Robotiq Hand-E OBJ meshes in `assets/robotiq_hande_description`, and local
RealSense D435i MJCF meshes in `assets/realsense_d435i` when present.
It also spawns `assets/ti_tray/ti_tray.obj` on the table if that mesh exists.
See [assets/README.md](assets/README.md) for source and license notes.

## Limitations

- Three points are the minimum but are sensitive to measurement noise.
- More than three contact points are recommended.
- Contact point offset calibration matters.
- Perpendicular contact helps data quality, but the plane estimate mainly uses contact positions.
- The fallback robot visualization is a clean geometric sketch, not a collision-accurate URDF model.
