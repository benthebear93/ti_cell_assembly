# Surface Estimator UR5e

Tools for UR5e + Robotiq Hand-E cell assembly: marker-relative grasping, pin
alignment and insertion, RealSense capture, and static/animated `viser` previews.
The package also estimates planes from measured contact poses and exports MuJoCo scenes.

All commands below run from **`surface_estimator_ur5e/`**, the directory containing
`pyproject.toml`. From the repository root, run `cd surface_estimator_ur5e` first.
Python 3.11 or newer is required. ROS is not required.

| Workflow | Start here |
| --- | --- |
| Plan or execute assembly on the robot | [Marker-based assembly](#run-marker-based-assembly) |
| Animate assembly without robot hardware | [Offline assembly preview](#preview-the-full-assembly-program-offline) |
| Inspect marker, holder, and TCP frames | [Static assembly visualization](#static-assembly-visualization) |
| Find calibrated coordinates and their units | [Calibration coordinates](#inspect-calibration-coordinates) |
| Change settings or find the relevant code | [Code and configuration map](#code-and-configuration-map) |
| Estimate a plane from saved contact poses | [Contact YAML](#contact-yaml-format) and [estimation](#run-estimation) |
| Export or run a simulated scene | [MuJoCo export](#export-a-mujoco-scene) |

## Install

### With uv

`uv sync` installs the project and the PyRoki Git revision declared in
[`pyproject.toml`](pyproject.toml), including the dependencies for offline IK:

```bash
uv sync
uv run python scripts/assembly_task.py --help
```

### With pip

Pip does not use `[tool.uv.sources]`, so install the same pinned PyRoki source explicitly:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "pyroki @ git+https://github.com/chungmin99/pyroki.git@388e43e1fc0d0ee382968d3dd72970fd62a0450c"
python -m pip install -e .
```

For dependency-only local module execution, `python -m pip install -r requirements.txt`
installs the base tools; that file does not include PyRoki, which `preview-motion`
requires. The editable install above also exposes the `surface-estimator` command.

## Assembly Command Overview

`assembly_task.py` dispatches to the script for each stage. Inspect its options with:

```bash
uv run python scripts/assembly_task.py marker-target --help
```

| Stage | Purpose | Hardware behavior |
| --- | --- | --- |
| `marker-target` | Plan or execute a selected assembly task | RTDE connection; movement/gripper commands require `--execute` |
| `preview-motion` | Animate or solve the complete assembly plan | No robot, gripper, or camera connection |
| `visualize` | Inspect saved marker/holder geometry and TCP targets | Reads RTDE and sets TCP by default; offline state can be supplied |
| `holder-align` | Align an already held tray from the current TCP | RTDE connection; motion requires `--execute` |
| `read-marker` | Save an ArUco pose in the robot base frame | RealSense + RTDE; sets TCP by default |
| `capture-realsense` | Save color/depth frames and metadata | RealSense |
| `image-align-sweep` | Capture images while adjusting local TCP pose | RealSense + RTDE; motion requires `--execute` |
| `run-sequence` | Run a recorded joint sequence | Dry run is offline; RTDE/gripper connections and motion require `--execute` |

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

For your measured contact file:

```bash
surface-estimator estimate --input data/ur5e_hande_3_surface_contacts.yaml
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

For your measured contact file:

```bash
surface-estimator visualize --input data/ur5e_hande_3_surface_contacts.yaml
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

## Run Marker-Based Assembly

Choose the extent of the motion with `--task`; the default is `pick`, and motion
requires `--execute`. Calibration and detailed stage settings are YAML overrides.

| Task | Last stage |
| --- | --- |
| `pick` | Marker-relative grasp |
| `align` | Grasp, four-pin rotation, and pin approach |
| `insert` | Four-pin insertion, keeping the gripper closed |
| `assembly` | Insertion, two-pin pickup/placement, and final tail motions |

From the project directory, inspect the full plan using the robot's calibrated
FK/IK without sending move or gripper commands:

```bash
uv run python scripts/assembly_task.py marker-target --task assembly
```

This dry run still connects to RTDE and sets the calibrated TCP by default.
For a completely offline preview, use `preview-motion` below. Execute the plan with:

```bash
uv run python scripts/assembly_task.py marker-target --task assembly --execute
```

Calibration, gripper settings, stage switches, and safety limits are defined in
[`scripts/assembly_config.py`](scripts/assembly_config.py), in `MotionConfig`.
Override only the values you need in a YAML mapping.
[`data/assembly_motion.yaml`](data/assembly_motion.yaml) starts as `{}` so defaults
are not duplicated. Replace `{}` with the settings you want to change, for example:

```yaml
speed_m_s: 0.02
acceleration_m_s2: 0.04
```

```bash
uv run python scripts/assembly_task.py marker-target \
  --task assembly --config data/assembly_motion.yaml
```

The same `--config` file works with `preview-motion`; the static `visualize`
command retains its own CLI settings. Effective values are `MotionConfig`
defaults, then YAML overrides, then explicit `--speed-m-s` / `--acceleration-m-s2`
overrides. Task selection, resume mode, robot IP, marker path, and `--execute`
remain CLI options. Unknown keys, wrong types, non-finite numbers, and invalid
limits are rejected before connecting to the robot.
Use `--from-current` to omit the initial joint move, or select one continuation:

```bash
uv run python scripts/assembly_task.py marker-target --task assembly --resume pins
uv run python scripts/assembly_task.py marker-target --resume release
```

`--resume pins` starts at pin approach and follows the selected task's stopping
point. The other resume choices are `release`, `after-close`, `tail`, and
`tail-next`; they run only that section from the measured TCP, without the initial
joint move or marker grasp. Add `--execute` to send the commands.

Migration from the previous CLI:

- `--align-to-four-pin-frame` becomes `--task align`.
- `--align-to-four-pin-frame --insert-after-pin-approach` becomes `--task assembly`.
- Add `--task insert` instead when insertion should finish without release/retreat.
- `--no-start-from-initial-pose` / `--skip-initial-pose` becomes `--from-current`.
- `--continue-…-from-current` becomes the corresponding `--resume` choice above.
- Other calibration options move to YAML with underscores, e.g. `--offset-x-mm 144`
  becomes `offset_x_mm: 144`. Boolean switches use explicit `true` / `false` values;
  e.g. `--no-post-two-pin-close-image-align` becomes `post_two_pin_close_image_align: false`.
- The empty `tail-extra` continuation and the unused `post_two_pin_after_close_final_y_mm`
  setting were removed, including the static viewer's unused matching CLI option.

## Code And Configuration Map

| File in `scripts/` | Responsibility |
| --- | --- |
| [`assembly_task.py`](scripts/assembly_task.py) | Dispatch stage commands |
| [`assembly_config.py`](scripts/assembly_config.py) | `MotionConfig` defaults, YAML overrides, and validation |
| [`assembly_plan.py`](scripts/assembly_plan.py) | Target poses, clearance calculations, and IK preflight |
| [`assembly_sequence.py`](scripts/assembly_sequence.py) | Shared preview list of moves, gripper actions, and dwells |
| [`marker_based_motion.py`](scripts/marker_based_motion.py) | Robot/gripper/camera execution through the `Robot` session |
| [`visualize_marker_frame.py`](scripts/visualize_marker_frame.py) | Static frames and paths drawn from the shared sequence |
| [`preview_assembly_motion.py`](scripts/preview_assembly_motion.py) | Offline IK sampling, part attachment, and animation |

Start reading execution at `Robot.marker_task`: it calls `align`, `pin_sequence`,
`post_insert`, and `aligned_release` in assembly order. The session owns robot
interfaces and settings; live execution recalculates from measured TCP poses at
action boundaries. Both previews use `assembly_sequence.py` with saved image
corrections, and tray paths follow its attach/detach events.

`tcp_shift(pose, x=…, y=…, z=…)` expresses local moves in millimetres; the common
pin-target calculation handles centering, approach, and insertion. Each setting's
range is declared beside its default in `MotionConfig`.

Shared helpers live in `src/surface_estimator_ur5e/` and are used by assembly,
standalone alignment, camera capture, and visualization:

| Module | Responsibility |
| --- | --- |
| `transforms.py` | UR pose conversion, local TCP offsets/rotations, point transforms |
| `workcell_geometry.py` | Holder pin frame, tray geometry, floor clearance, shared workcell defaults |
| `calibration.py` | Saved marker YAML and tool-to-camera calibration XML |
| `robot_io.py` | Lazy RTDE connections, TCP reads and offset setting |

These helpers take poses, paths, and values directly. The motion planner no
longer imports the visualization script. UR poses use metres and rotation
vectors in radians; tray geometry takes 4×4 transforms. The separate joint
sequence retains its own TCP calibration.

Configuration and model sources:

| Information | Authoritative source |
| --- | --- |
| Assembly stage defaults | `MotionConfig` in `scripts/assembly_config.py`; YAML contains overrides only |
| Assembly TCP, initial pose, holder/tray placement | `workcell_geometry.py` |
| URDF description name, joint and actuator order | `robot_model.py` |
| Marker measurement | `data/markers/aruco_364_in_base.yaml` |
| Hand-eye transform and camera intrinsics | `data/calibration/realsense_ur5e/Calibration.xml` and `camera_intrinsics.yaml` |
| Physical robot FK/IK | UR controller, via RTDE |
| Offline FK/IK | The `ur5e_description` URDF loaded through `robot_descriptions` |
| Approximate visualization fallback | DH parameters in `SimpleUR5eVisualizer` |
| MuJoCo robot model | `assets/universal_robots_ur5e/ur5e.xml` |

The XML files in `models/` are generated scenes: a fixed visualization, an
actuated robot, and a tray/gripper simulation. They contain exported model data
and are not independent calibration sources. The two motion-sequence YAML files
contain different recorded trajectories. Saved targets, contact files, and image
metadata retain the settings used when they were recorded.

The joint-sequence TCP remains `[0, 0, 0.152, 0, 0, 0]`, while assembly uses
`[0, 0, 0.158, -1.5707, 0, 0]`. The static `visualize` command also retains its
existing tail-stage offsets, which differ from execution defaults. Use
`preview-motion` for playback of the current assembly settings. These differences
must be resolved against the actual setup before treating the modes as equivalent.

## Inspect Calibration Coordinates

For the marker's position and orientation in the **robot base frame**, open
[`data/markers/aruco_364_in_base.yaml`](data/markers/aruco_364_in_base.yaml).
If the command uses `--marker-pose`, inspect that file instead.

| Field under `pose_in_base` | Meaning |
| --- | --- |
| `translation_m` | Marker-center X, Y, Z in metres |
| `rotation_rpy_deg` | Saved roll, pitch, yaw in degrees |
| `rotation_quaternion_xyzw` | The same orientation as an XYZW quaternion |
| `transform_matrix` | 4×4 transform mapping marker coordinates into robot-base coordinates; last column contains translation in metres |

The loader uses `transform_matrix` when present. The other fields are alternate
representations of the saved pose, not separate calibrations; changing only
`translation_m` or `rotation_rpy_deg` will not override an existing matrix.
The bundled measurement currently has X = **−80.046 mm**, Y = **−467.841 mm**,
Z = **997.781 mm**. Read the YAML for the full precision and any newer capture;
`source.captured_at` and `capture_quality` identify the capture and its spread.

Other calibration and placement values:

| Information | File and field |
| --- | --- |
| Camera pose expressed in the calibrated TCP/tool frame | [`Calibration.xml`](data/calibration/realsense_ur5e/Calibration.xml), `CalibrationResult / MovingTransform frame="Camera"`: `Vector3D` in metres and `Rotation3D/Rotation3D` |
| Camera intrinsics used for marker detection | [`camera_intrinsics.yaml`](data/calibration/realsense_ur5e/camera_intrinsics.yaml) |
| Default assembly TCP offset | [`workcell_geometry.py`](src/surface_estimator_ur5e/workcell_geometry.py), `DEFAULT_TCP_OFFSET_UR`: XYZ in metres, rotation vector in radians |
| Default holder placement relative to the marker | The same file's `DEFAULT_ASSEMBLY_ORIGIN_*_MM`, `DEFAULT_ASSEMBLY_RPY_DEG`, and `DEFAULT_ASSEMBLY_LOCAL_YAW_DEG` |
| Overrides used by a particular assembly run | The YAML passed to `--config`, for example [`data/assembly_motion.yaml`](data/assembly_motion.yaml); omitted settings use `MotionConfig` defaults |

The current marker pose comes from the marker YAML above. The XML's
`StaticTransform frame="Marker"` belongs to the hand-eye calibration run.

To compare the original measurement with the floor-constrained frame visually:

```bash
uv run python scripts/assembly_task.py visualize \
  --show-raw --show-holder --show-plan-frames
```

The default `floor` mode keeps the measured marker position but constrains its
orientation to the horizontal floor. Robot-base +Z points physically downward
in this ceiling-mounted setup. `--show-raw` also draws the original measured
frame; it does not change the frame used for target calculation. Use
`--marker-frame-mode raw` to calculate targets from the measured orientation.

This visualization reads robot state and sets the configured TCP by default,
without sending motion commands. See [static visualization](#static-assembly-visualization)
for supplying an offline robot state.

## Regression Checks

Hardware-free regression tests compare motion targets, speeds, gripper commands,
and dwell times against saved traces, including measured TCP offsets and stage
continuations. Preview tests also compare static frames, interpolated TCP/tray
paths, complete tail stages, and camera coordinates. See
[`tests/fixtures/README.md`](tests/fixtures/README.md) for baseline provenance.

```bash
uv run --with pytest python -m pytest -q tests
uv run --with ruff ruff check scripts src tests
```

With an activated environment containing `pytest`, use `python -m pytest -q tests`.
The test doubles replace robot/gripper connections; they do not establish physical
reachability or collision clearance.

## Static Assembly Visualization

Inspect the saved marker and holder against the robot's current TCP, including
the optional insertion and later stage frames:

```bash
uv run python scripts/assembly_task.py visualize \
  --show-holder --show-tray --show-plan-frames --insert-after-pin-approach
```

This command reads robot state and sets the assembly TCP by default; it sends no
motion commands. Adding `--show-motion-trajectory` uses controller FK/IK to draw
the sampled path. Supplying both `--q-deg` and `--tcp-pose-ur` allows static frame
inspection without RTDE when `--show-motion-trajectory` is omitted.
Use `preview-motion` below for a completely offline animated trajectory using
the current assembly settings; the static viewer retains its legacy tail offsets.

## Preview The Full Assembly Program Offline

Animate the Cartesian targets and gripper events used by
`scripts/marker_based_motion.py` without connecting to the UR5e or Hand-E:

```bash
uv run python scripts/assembly_task.py preview-motion
```

The default task is `assembly`; `--task align` and `--task insert` stop earlier.
To use the same overrides as the execution command:

```bash
uv run python scripts/assembly_task.py preview-motion --config data/assembly_motion.yaml
```

This mode starts from the saved initial pose and saved image corrections.
It does not accept `--execute`, `--from-current`, `--resume`, `--task pick`, or
live camera search (`auto_pin_image_align: true`). The first URDF load may
download and cache the robot description even though no robot hardware is used.

Then open the URL printed by `viser`, normally `http://localhost:8080`. The
preview starts from the saved initial joint pose, uses the saved marker and image
alignment metadata, and includes marker grasp, tray lift, four-pin rotation,
pin approach/insertion, release, adjacent two-pin motion, and the final tail
motions. PyRoki solves every interpolated TCP pose against the same UR5e URDF
used by the visualizer. The preview renders `assets/ti_tray_short.stl` (converted
from millimetres to metres), extracts its four hole centers, and fits them to the
four holder-pin axes at insertion. Its remaining pin-axis degree of freedom is
chosen so the initial tray bottom sits on the marker plane; the resulting rigid
TCP-to-tray attachment is kept throughout the grasp and carry. The Hand-E
preview is rooted at `tool0` and uses lightweight envelopes of the repository's
Hand-E meshes. The meter-scale
`assets/picknik_ur5_realsense_camera_adapter_rev2.STL` is centered on the UR
flange bolt pattern, and the Hand-E visual is shifted 7 mm along flange Z to sit
on its upper mounting face. These are visualization-only changes: the real motion geometry,
RTDE path, and gripper commands are unchanged, and no RTDE or gripper socket is
opened. The lower handle is shortened from 20 mm to 5 mm in the rendered mesh
so it meets the saved grasp pose; override this with
`--visual-tray-handle-scale` if needed. The STL on disk is not modified.

The later two-pin sequence also renders `assets/CATHODE_PLATE_w_handle.stl`.
Its two large holes nearest the handle start on the holder's adjacent two pins.
The plate underside is seated on the holder support surface at the start of the
orange pin sections. The plate attaches at `post-insert two-pin close`, follows
the existing lift and transfer path, seats its four large holes on the four-pin
datum, and detaches at `post-two-pin aligned open`. The small saved
image-alignment correction is blended into the visualization after the plate has
lifted clear of the two pins.

Useful playback options:

```bash
uv run python scripts/assembly_task.py preview-motion \
  --time-scale 6 \
  --playback-hz 20 \
  --no-loop
```

Use the `Motion path` checkbox in the viser controls to hide or restore the
colored Cartesian path lines while the preview is running.

To preflight the complete trajectory without starting a browser server:

```bash
uv run python scripts/assembly_task.py preview-motion --plan-only
```

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
