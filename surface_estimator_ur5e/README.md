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

## Preview The Full Assembly Program Offline

Animate the Cartesian targets and gripper events used by
`scripts/marker_based_motion.py` without connecting to the UR5e or Hand-E:

```bash
uv run python scripts/assembly_task.py preview-motion
```

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

## Train DMPs For The Complete Assembly Motion

Train one Cartesian DMP for every continuous `moveL` stage in the full real
assembly sequence:

```bash
uv run python scripts/assembly_task.py train-all-motion-dmps \
  --output data/dmp/all_assembly_motion_dmps.npz
```

This is fully offline. It reconstructs the same command targets as
`marker_based_motion.py`, validates the complete nominal path with strict
PyRoki IK, and trains 27 ordered move primitives. The eight gripper/dwell
commands remain discrete events in the same 35-event program. It never opens
an RTDE, gripper, or camera connection.

The NPZ stores position and quaternion-DMP weights, start/goal poses, durations,
training seeds, and the event-to-primitive mapping. The JSON beside it records
the ordered program and reproduction errors. As a final check, the trainer
loads only the saved weights and endpoints into fresh DMP instances, rolls out
all 27 motions without calling `train()`, and compares them with the trained
models. Generated files under `data/dmp/` are ignored by Git.

Replay every saved primitive for a complete nominal task, return the same robot
to the shared start, reset the visual workpieces, and then replay the complete
task at a marker-frame goal offset:

```bash
uv run python scripts/assembly_task.py compare-all-motion-dmp \
  --offset-marker-mm 20 -200 0
```

Add a third complete task at marker offset `(X -250, Y +150, Z 0) mm` in the
same one-robot viewer:

```bash
uv run python scripts/assembly_task.py compare-all-motion-dmp \
  --offset-marker-mm 20 -200 0 \
  --second-offset-marker-mm -250 150 0
```

The sequence is nominal, first offset, then second offset, with a return to the
shared robot start and newly introduced tray/cathode workpieces between tasks.
The second task completes both part insertions and the original post-release
tail at `(-250, 150, 0)`. The placement leaves the holder and inserted
workpieces physically disjoint. A conservative dense swept-volume check over
the nominal/first-offset DMP path gives 49.5 mm minimum AABB clearance between the
second holder and the UR5e collision meshes plus the flange-adapter/Hand-E
envelope. The originally tested `(40, -400, 0)` placement has no strict IK
solution for the empty-gripper X-rotation after release; `(-250, 150, 0)` keeps
the full unmodified 27-stage Cartesian program inside the verified IK region.

Each task uses all 27 saved move DMPs plus the original eight gripper/dwell
events. The offset half rebuilds all holder-relative targets before changing
each primitive start/goal; it does not retrain the weights. Both full paths are
strict-IK checked before the one-robot, original-material Viser scene starts.
Nominal tray/cathode objects remain at their completed holder poses; the offset
half introduces a separate tray and cathode plate instead of teleporting or
reusing the nominal workpieces.

Translation and holder-frame yaw can be retargeted together. For example,
shift the second task by marker X `+200 mm` and rotate its complete holder frame
by marker Z `+90°`:

```bash
uv run python scripts/assembly_task.py compare-all-motion-dmp \
  --offset-marker-mm 200 0 0 \
  --offset-yaw-deg 90
```

The X `+200 mm`, marker-Z `+180°` offset lies on the edge of the UR5e workspace
with the nominal TCP-to-part grasp. Since the nominal holder origin starts at
X `-60 mm`, the resulting offset holder origin is marker X `+140 mm`. Use the
Hand-E's equivalent local-Y-flipped grasp, move the grasp point 10 mm inward,
and require a 50 mm post-grasp lift:

```bash
uv run python scripts/assembly_task.py compare-all-motion-dmp \
  --offset-marker-mm 200 0 0 \
  --offset-yaw-deg 180 \
  --post-grasp-lift-mm 50 \
  --offset-equivalent-grasp-y-flip \
  --offset-grasp-shift-z-mm 10
```

This keeps the tray/cathode pin goals fixed. Only the visualization-only grasp
gauge changes. If a retargeted DMP curve leaves the strict-IK region, the viewer
projects that primitive toward its endpoint-equivalent `moveL` path and reports
the retained DMP-shape percentage. It still never changes or executes the real
robot controller program.

## Generate Tray Pick/Insert DMP Demonstrations

Fit three Cartesian DMP primitives (grasp approach, free-space transfer, and
constrained insertion) to the real tray motion prefix and generate synthetic
demonstrations offline:

```bash
uv run python scripts/assembly_task.py generate-tray-dmp \
  --episodes 20 \
  --initial-joint-noise-deg 5 \
  --output data/dmp/tray_pick_insert_dmp.npz
```

The generator imports `CartesianDMP` from `~/simple_dmp`, keeps gripper
close/open as discrete phases, and fixes every generated grasp and insertion
endpoint to the original motion. The first episode uses the nominal initial
joint pose. Later episodes independently sample each initial joint within the
requested half-range, use FK for the matching initial TCP, and adapt the grasp
approach DMP from that start. Free-space DMP weights and episode timing are also
varied; insertion weight noise is disabled by default. Every candidate is
checked with strict PyRoki IK, joint limits, initial-TCP displacement limits,
branch continuity, joint speed, and the tray-to-four-pin endpoint fit before it
is accepted.

The compressed NPZ contains episode boundaries, time, TCP and joint states,
gripper state, phase IDs, next-pose actions, and local TCP delta actions. A JSON
file beside it records DMP reproduction errors, variation amounts, rejection
reasons, and validation results. Generated `data/dmp/` files are ignored by
Git. These are kinematic synthetic trajectories: collision/contact physics,
camera observations, and real-robot success are deliberately not claimed.
Schema version 2 additionally stores per-episode initial joint positions, FK
TCP poses, and joint offsets as `episode_initial_*` arrays.

Retarget the trained transfer/insertion DMP goals by a fixed marker-frame
translation without retraining their nominal weights:

```bash
uv run python scripts/assembly_task.py generate-tray-dmp \
  --goal-offset-marker-mm 50 0 0 \
  --episodes 20 \
  --output data/dmp/tray_pick_insert_dmp_xplus50.npz
```

The offset is stored in JSON metadata. `preview-tray-dmp` reads it automatically
and moves the visual holder/pins by the same marker-frame translation, so no
extra viewer offset argument is needed.

Generate goal-conditioned recovery demonstrations from those base episodes:

```bash
uv run python scripts/assembly_task.py generate-tray-dmp-recovery \
  --input data/dmp/tray_pick_insert_dmp.npz \
  --recoveries-per-episode 2 \
  --output data/dmp/tray_pick_insert_dmp_recovery.npz
```

For each base episode, the recovery generator varies the selected free-space
phase start and goal, rolls out a nominal DMP, perturbs one intermediate
position/orientation, and trains a suffix DMP back to the varied goal. The
output stores both the nominal path and ragged recovery paths. Recovery samples
include PyRoki-FK state, next TCP/joint command, local next-action delta, and a
direct local residual to the conditioned goal. Goal variation defaults to the
`transfer_to_preinsert` phase only; it does not alter the base dataset's exact
four-pin insertion endpoint.

This remains kinematic data. “Realized state” means strict-IK joints evaluated
again with PyRoki FK; controller lag, dynamics, contact, MuJoCo, and the real
robot are not involved.

Inspect a generated dataset in the offline Viser viewer:

```bash
uv run python scripts/assembly_task.py preview-tray-dmp \
  --dataset data/dmp/tray_pick_insert_dmp.npz
```

The viewer overlays every generated TCP path in gray and colors the selected
episode by DMP phase. Use **Episode** to switch demonstrations, **Play** and
**Playback multiplier** to animate them, and the two path checkboxes to hide or
show the overlays. Initial TCP markers and the **Start joint**/**Start TCP**
fields make the randomized starting-state spread explicit. The robot follows
the saved joint trajectory while the tray is attached and is left seated on
the four holder pins after release. This viewer never opens an RTDE or gripper
connection.

Preview a single robot executing the nominal task and then moving that same tray
from the nominal pins to the retargeted goal:

```bash
uv run python scripts/assembly_task.py compare-tray-dmp \
  --nominal-dataset data/dmp/tray_pick_insert_dmp_initial6.npz \
  --offset-dataset data/dmp/tray_pick_insert_dmp_xplus20_yminus50.npz
```

The viewer first plays the complete nominal pick/insert/release. Without
resetting the robot or tray, it closes the gripper again at the nominal inserted
pose, reverses the constrained insertion to lift the tray clear of the pins,
uses a strict-IK moveL bridge between the two pre-insert poses, and then plays
the offset insertion/release. The robot and workcell keep their original
materials; this remains a read-only offline visualization.

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
