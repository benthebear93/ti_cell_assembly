# Robot Assets

The package runs without local URDF files by using `robot_descriptions.py` to fetch the `ur5e_description` model on demand. If that fails, it falls back to `SimpleUR5eVisualizer`, a lightweight line-and-frame visualization with a small Hand-E style gripper sketch.

For future URDF visualization, place robot assets here without hard-coding absolute paths. A suggested layout is:

```text
assets/
├── urdf/
│   ├── ur5e_hande.urdf
│   └── meshes/
│       └── ...
└── README.md
```

Good sources are:

- Universal Robots UR description packages for UR5e geometry: https://github.com/UniversalRobots/Universal_Robots_ROS2_Description
- `robot_descriptions.py` UR5e package: https://github.com/robot-descriptions/robot_descriptions.py
- Robotiq Hand-E description package: https://github.com/macmacal/robotiq_hande_description

Make sure mesh paths inside the URDF are relative to the URDF location or otherwise resolvable by `yourdfpy`.

The MuJoCo exporter currently looks for local assets at:

- `assets/universal_robots_ur5e/ur5e.xml`
- `assets/robotiq_hande_description/meshes/io_coupler.obj`
- `assets/robotiq_hande_description/meshes/hande.obj`
- `assets/robotiq_hande_description/meshes/finger.obj`
- `assets/ti_tray/mujoco_parts/ti_tray_part_*.obj`

The Hand-E OBJ files were converted from the upstream DAE meshes because MuJoCo
loads OBJ meshes directly. The upstream Apache-2.0 license is kept at
`assets/robotiq_hande_description/LICENSE`.

The TI tray is split into connected-component OBJ parts for MuJoCo. Loading the
original `assets/ti_tray/ti_tray.obj` as one mesh caused MuJoCo to simplify it
into a small hull, losing the visible plate shape.
