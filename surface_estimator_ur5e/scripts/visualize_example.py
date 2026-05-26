"""Launch the example visualization."""

from pathlib import Path

from surface_estimator_ur5e.geometry import estimate_plane_from_points
from surface_estimator_ur5e.io import load_contact_file
from surface_estimator_ur5e.visualization import launch_visualization


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    contact_data = load_contact_file(repo_root / "data" / "example_contacts.yaml")
    normal_hint = (
        contact_data.normal_direction_vector
        if contact_data.normal_direction_vector is not None
        else contact_data.normal_direction
    )
    plane = estimate_plane_from_points(
        contact_data.contact_points(),
        normal_hint=normal_hint,
        flip=contact_data.flip_normal,
    )
    launch_visualization(contact_data, plane)


if __name__ == "__main__":
    main()
