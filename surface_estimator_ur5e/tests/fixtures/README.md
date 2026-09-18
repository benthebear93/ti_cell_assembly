`marker_motion_before.json` records the real command order from
`scripts/marker_based_motion.py` at commit
`feac9facaa8cb1d264101aa93f05b08c20e49811`, before the refactor.

The capture used the robot and gripper doubles in `../motion_fakes.py`, the
repository's marker/holder/tray geometry, and saved image corrections. It covers
18 stage configurations, both with perfect TCP tracking and with a fixed
measurement error after every moveL. The two live-camera cases replace the image
search with a deterministic measured correction; no camera or robot was opened.

Targets, speed/acceleration, blocking mode, gripper settings, dwell times, and
connection cleanup are compared. IK returns the same safe joint branch for each
reachable pose; physical reachability and collisions are outside these tests.
The `resume pins` cases start at the recorded full assembly's four-pin alignment
pose. Other cases start at `INITIAL_POSE` in the doubles.

Do not regenerate these expected traces from the refactored implementation:
that would discard the independent behavior baseline.

`camera_search_before.json` comes from the same original revision. It records
the actual search loop with mocked frames/scores: one case improves throughout
both iterations, and one retains the first iteration's best image. Candidate
order, poses, speeds, settle times and the saved best image are checked without
opening camera hardware or writing captures.

`alignment_motion_before.json` records `align_tray_to_pins_rotation.py` before
the shared-helper extraction (SHA256
`ec1a006aceb7991feb35a223bd10fd16c6612b5c02812a74c7662f1029010286`).
The capture loaded snapshots of both that script and its original viewer helpers.
It uses the same robot/gripper doubles for dry run, lift/rotation, gripper close,
TCP-setting bypass, and rotation/lift limits. No hardware was connected.

`visual_motion_before.json` was captured immediately before extracting the shared
preview sequence. Its `sha256` entries identify the source snapshots. Seven
static-viewer cases record endpoint transforms, sample counts, full TCP and tray
paths, and motion frames; five animation cases record every move, gripper action,
attachment change, and dwell. They use the same geometry and `FakeRobot` as above.
The old static viewer's full tail raised `ValueError: too many values to unpack`;
the repaired full trajectory is checked against the independently recorded
animated assembly sequence with matching settings. No hardware was connected.
