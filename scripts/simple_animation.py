"""Working PyVista animation using BackgroundPlotter."""

import numpy as np
import pyvista as pv

try:
    from pyvistaqt import BackgroundPlotter

    HAS_PYVISTAQT = True
except ImportError:
    print("pyvistaqt not installed. Install with: uv pip install pyvistaqt")
    HAS_PYVISTAQT = False
    import sys

    sys.exit(1)

print("Creating background plotter...")
plotter = BackgroundPlotter()

# Create a simple sphere
sphere = pv.Sphere()
actor = plotter.add_mesh(
    sphere, scalars=np.zeros(sphere.n_points), cmap="viridis", clim=[0, 1]
)

# Add text
text = plotter.add_text("Frame: 0", position="upper_left", font_size=12)

# Animation state
frame = [0]


def update_frame():
    """Update the animation frame."""
    frame[0] = (frame[0] + 1) % 100

    # Update scalars
    scalars = np.sin(frame[0] / 10.0 + np.arange(sphere.n_points) / 10.0)
    sphere.point_data["scalars"] = scalars

    # Update text
    text.SetText(2, f"Frame: {frame[0]}")

    print(f"Frame {frame[0]}")


# Add callback that runs repeatedly
plotter.add_callback(update_frame, interval=50)  # 50ms = ~20 FPS

print("Animation started. Close window to exit.")
print("Press Ctrl+C in terminal to stop.")

# Keep the script running - BackgroundPlotter needs the event loop
try:
    import sys

    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        sys.exit(app.exec_())
except KeyboardInterrupt:
    print("\nStopped by user")

print("Animation window closed.")
