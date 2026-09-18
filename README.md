# TI Cell Assembly

UR5e + Robotiq Hand-E cell assembly, with marker/camera calibration, offline
motion previews, contact-based surface estimation, and MuJoCo tooling.
The Python project, scripts, settings, and assets live in `surface_estimator_ur5e/`.

Use the project from that directory:

```bash
cd surface_estimator_ur5e
uv sync
uv run python scripts/assembly_task.py --help
```

Start with the [offline assembly preview](surface_estimator_ur5e/README.md#preview-the-full-assembly-program-offline),
or see [robot execution and configuration](surface_estimator_ur5e/README.md#run-marker-based-assembly).
`marker-target` dry runs still connect to the robot; `preview-motion` uses local FK/IK.

The [project guide](surface_estimator_ur5e/README.md) also covers
[calibrated coordinates](surface_estimator_ur5e/README.md#inspect-calibration-coordinates),
[code and configuration ownership](surface_estimator_ur5e/README.md#code-and-configuration-map),
[surface estimation](surface_estimator_ur5e/README.md#contact-yaml-format), and
[MuJoCo export](surface_estimator_ur5e/README.md#export-a-mujoco-scene).
