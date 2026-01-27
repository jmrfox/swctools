import numpy as np
import pyvista as pv
import logging

# Try to import BackgroundPlotter for working animations
try:
    from pyvistaqt import BackgroundPlotter

    HAS_BACKGROUND_PLOTTER = True
except ImportError:
    HAS_BACKGROUND_PLOTTER = False


def _frusta_to_pyvista_meshes(frusta_set):
    """Convert a FrustaSet into a list of individual PyVista meshes.

    Each frustum becomes one UnstructuredGrid with a single polyhedron cell,
    preserving the per-frustum structure needed for per-compartment voltage coloring.

    Parameters
    ----------
    frusta_set : FrustaSet
        The batched frusta mesh.

    Returns
    -------
    list of pyvista.UnstructuredGrid
        One mesh per frustum, each containing a single polyhedron cell.
    """
    from .geometry import frustum_mesh

    logger = logging.getLogger(__name__)

    logger.info(
        "Converting FrustaSet to PyVista meshes: n_frusta=%d sides=%d end_caps=%s",
        frusta_set.n_frusta,
        frusta_set.sides,
        frusta_set.end_caps,
    )

    meshes = []
    for frustum in frusta_set.frusta:
        vertices, faces = frustum_mesh(
            frustum, sides=frusta_set.sides, end_caps=frusta_set.end_caps
        )

        # Convert to numpy arrays
        points = np.array(vertices, dtype=np.float32)

        # Build a single polyhedron cell from all faces
        # Format for polyhedron: [cell_size, n_faces, n_pts_face0, pt0, pt1, ..., n_pts_face1, pt0, pt1, ...]
        face_data = []
        for face in faces:
            face_data.append(3)  # triangular face
            face_data.extend(face)

        # Cell size = 1 (n_faces) + len(face_data)
        cell_size = 1 + len(face_data)
        cell = [cell_size, len(faces)] + face_data

        cells = np.array(cell, dtype=np.int32)
        cell_types = np.array([pv.CellType.POLYHEDRON], dtype=np.uint8)

        # Create UnstructuredGrid with one polyhedron cell
        mesh = pv.UnstructuredGrid(cells, cell_types, points)
        meshes.append(mesh)

        if (len(meshes) % 10 == 0) or (len(meshes) == frusta_set.n_frusta):
            logger.debug("Created %d/%d meshes", len(meshes), frusta_set.n_frusta)

    logger.info("Completed mesh conversion: created %d meshes", len(meshes))
    return meshes


def animate_compartment_voltage(
    time_domain,
    voltage_traces,
    segment_meshes,
    *,
    cmap="viridis",
    clim=None,
    scalar_bar=True,
    scalar_bar_args=None,
    fps=30,
    stride=1,
    window_size=(900, 700),
    background="white",
    show_axes=False,
    notebook=False,
    off_screen=False,
    movie_path=None,
    interpolation="nearest",
    use_background_plotter=None,
):
    """
    Animate voltage over a multi-compartment neuron using PyVista.

    Parameters
    ----------
    time_domain : (T,) array
        Time values corresponding to voltage samples.
    voltage_traces : (T, N) array
        Voltage time series for N compartments.
    segment_meshes : list of pyvista.PolyData
        One mesh per compartment. Each mesh should represent exactly
        one segment / compartment.
    cmap : str
        Matplotlib colormap name.
    clim : tuple or None
        Color limits (vmin, vmax). If None, inferred from voltage_traces.
    scalar_bar : bool
        Whether to show scalar bar.
    scalar_bar_args : dict or None
        Passed to add_scalar_bar().
    fps : int
        Frames per second for animation / movie.
    stride : int
        Temporal downsampling factor (use every `stride` time steps).
    window_size : tuple
        Render window size in pixels.
    background : str or tuple
        Background color.
    show_axes : bool
        Show orientation axes.
    notebook : bool
        If True, enables notebook-friendly rendering.
    off_screen : bool
        Enable off-screen rendering (required for movie export).
    movie_path : str or None
        If provided, renders animation to a movie file (e.g. "voltage.mp4").
    interpolation : {"nearest", "linear"}
        Temporal interpolation mode if stride > 1.
    use_background_plotter : bool or None
        If True, use BackgroundPlotter from pyvistaqt (requires pyvistaqt and Qt).
        If False, use standard Plotter (timer callbacks may not work on all systems).
        If None (default), automatically use BackgroundPlotter if available.

    Returns
    -------
    plotter : pyvista.Plotter or BackgroundPlotter
        The plotter instance (useful for interactive tweaking).

    Notes
    -----
    PyVista's standard Plotter has unreliable timer support on some systems.
    For working interactive animations, install pyvistaqt and a Qt backend:
        uv add --dev pyvistaqt PySide6
    If BackgroundPlotter is not available, the function will use standard Plotter
    but timer callbacks may not fire, resulting in a static display.
    """
    logger = logging.getLogger(__name__)

    # Determine which plotter to use
    if use_background_plotter is None:
        use_bg = HAS_BACKGROUND_PLOTTER
    else:
        use_bg = use_background_plotter
        if use_bg and not HAS_BACKGROUND_PLOTTER:
            logger.warning(
                "BackgroundPlotter requested but pyvistaqt not available. "
                "Install with: uv add --dev pyvistaqt PySide6"
            )
            use_bg = False

    logger.info(
        "Starting animate_compartment_voltage: fps=%d stride=%d window_size=%s use_background_plotter=%s",
        fps,
        stride,
        window_size,
        use_bg,
    )

    time_domain = np.asarray(time_domain)
    V = np.asarray(voltage_traces)

    logger.debug(
        "Input shapes: time_domain=%s voltage_traces=%s", time_domain.shape, V.shape
    )

    n_times, n_segments = V.shape
    assert len(time_domain) == n_times, (
        f"time_domain length ({len(time_domain)}) must match voltage_traces "
        f"first dimension ({n_times})"
    )
    assert len(segment_meshes) == n_segments, (
        f"Number of meshes ({len(segment_meshes)}) must match number of "
        f"voltage traces ({n_segments})"
    )

    # --- Temporal subsampling ---
    frame_indices = np.arange(0, n_times, stride)
    frame_times = time_domain[frame_indices]
    logger.info(
        "Animation frames: %d (from %d time steps with stride=%d)",
        len(frame_indices),
        n_times,
        stride,
    )

    # --- Merge geometry into a single mesh ---
    logger.info("Merging %d segment meshes...", len(segment_meshes))
    merged = segment_meshes[0].copy()
    for i, mesh in enumerate(segment_meshes[1:], 1):
        merged = merged.merge(mesh, merge_points=False)
        if (i % 10 == 0) or (i == len(segment_meshes) - 1):
            logger.debug("Merged %d/%d meshes", i + 1, len(segment_meshes))

    n_cells = merged.n_cells
    logger.info("Merge complete: n_cells=%d n_segments=%d", n_cells, n_segments)
    assert n_cells == n_segments, "Merged mesh must have exactly one cell per segment"

    # --- Initial scalar field ---
    scalars = V[frame_indices[0]].astype(np.float32)
    merged.cell_data["voltage"] = scalars

    # --- Color limits ---
    if clim is None:
        clim = (np.min(V), np.max(V))
    logger.info("Color limits: clim=%s", clim)

    # --- Plotter setup ---
    if use_bg:
        logger.info("Creating BackgroundPlotter (interactive animation will work)")
        plotter = BackgroundPlotter()
        if window_size:
            plotter.window_size = window_size
    else:
        logger.info(
            "Creating standard Plotter: notebook=%s off_screen=%s", notebook, off_screen
        )
        if not HAS_BACKGROUND_PLOTTER:
            logger.warning(
                "Using standard Plotter - timer callbacks may not work on your system. "
                "For working animations, install: uv add --dev pyvistaqt PySide6"
            )
        plotter = pv.Plotter(
            window_size=window_size,
            notebook=notebook,
            off_screen=off_screen,
        )

    plotter.set_background(background)
    logger.debug("Plotter created and configured")

    actor = plotter.add_mesh(
        merged,
        scalars="voltage",
        clim=clim,
        cmap=cmap,
        show_scalar_bar=scalar_bar,
    )

    if scalar_bar and scalar_bar_args is not None:
        plotter.add_scalar_bar(**scalar_bar_args)

    if show_axes:
        plotter.show_axes()

    # --- Time annotation ---
    time_text = plotter.add_text(
        f"t = {frame_times[0]:.3f}",
        position="upper_left",
        font_size=12,
    )

    # --- Animation state ---
    class AnimationState:
        def __init__(self):
            self.frame_idx = 0
            self.playing = True
            self.movie_file = movie_path

    state = AnimationState()

    # --- Movie setup ---
    if movie_path is not None:
        plotter.open_movie(movie_path, framerate=fps)

    # --- Animation callback ---
    def update_frame():
        """Update the mesh with the current frame's voltage data."""
        logger.debug(
            "update_frame() called - playing=%s frame_idx=%d",
            state.playing,
            state.frame_idx,
        )

        if not state.playing:
            return

        ti = frame_indices[state.frame_idx]

        if state.frame_idx % 10 == 0:
            logger.debug(
                "Rendering frame %d/%d (t=%.3f)",
                state.frame_idx,
                len(frame_indices),
                time_domain[ti],
            )

        # Update voltage data
        if interpolation == "nearest":
            scalars[:] = V[ti]
        else:
            scalars[:] = V[ti]

        merged.cell_data["voltage"] = scalars
        time_text.SetText(2, f"t = {time_domain[ti]:.3f}")

        if movie_path is not None and state.movie_file is not None:
            plotter.write_frame()

        # Advance to next frame
        state.frame_idx = (state.frame_idx + 1) % len(frame_indices)

        # Close movie at end if recording
        if (
            movie_path is not None
            and state.frame_idx == 0
            and state.movie_file is not None
        ):
            logger.info("Closing movie file: %s", movie_path)
            plotter.close()
            state.movie_file = None

    # --- Keyboard controls ---
    def toggle_play():
        """Toggle play/pause with spacebar."""
        state.playing = not state.playing
        status = "Playing" if state.playing else "Paused"
        logger.info(
            "Animation %s (frame %d/%d)", status, state.frame_idx, len(frame_indices)
        )

    def reset_animation():
        """Reset to first frame with 'r' key."""
        state.frame_idx = 0
        state.playing = False
        ti = frame_indices[0]
        scalars[:] = V[ti]
        merged.cell_data["voltage"] = scalars
        time_text.SetText(2, f"t = {time_domain[ti]:.3f}")
        plotter.render()
        logger.info("Animation reset to frame 0")

    # Add keyboard callbacks
    plotter.add_key_event("space", toggle_play)
    plotter.add_key_event("r", reset_animation)

    # Add instructions text
    instructions = plotter.add_text(
        "Controls: [Space] Play/Pause  [R] Reset  [Q] Quit",
        position="lower_left",
        font_size=10,
        color="gray",
    )

    # --- Start animation ---
    logger.info("Starting interactive animation with %d frames", len(frame_indices))
    logger.info("Controls: [Space] Play/Pause, [R] Reset, [Q] Quit")

    # Calculate timer interval in milliseconds
    timer_interval = int(1000 / fps)
    logger.debug("Setting up timer with interval=%d ms", timer_interval)

    if use_bg:
        # BackgroundPlotter uses add_callback which actually works
        plotter.add_callback(update_frame, interval=timer_interval)
        logger.debug("Callback added to BackgroundPlotter")
        logger.info("Opening animation window - animation will play automatically")

        # Keep the Qt event loop running
        try:
            import sys
            from qtpy.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                sys.exit(app.exec_())
        except KeyboardInterrupt:
            logger.info("Animation stopped by user")
    else:
        # Standard Plotter - use add_timer_event (may not work on all systems)
        plotter.add_timer_event(
            max_steps=len(frame_indices) * 1000,
            duration=timer_interval,
            callback=update_frame,
        )
        logger.debug("Timer event added")
        logger.warning(
            "Opening plotter window - animation may not play if timer callbacks don't work on your system"
        )
        plotter.show()

    logger.info("Animation window closed")
    return plotter
