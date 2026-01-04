"""Viser visualization utilities for RoboVerse demos.

This module provides a unified ViserVisualizer class for interactive 3D visualization
of robots, objects, and trajectories using the viser library.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import MISSING
from pathlib import Path
from typing import Any

import numpy as np
import torch
import viser




class ViserVisualizer:
    """Interactive 3D visualizer for robots and objects using viser.

    This class provides comprehensive visualization and control capabilities including:
    - Loading and displaying URDF models (robots and objects)
    - Primitive shape visualization (cubes, spheres, cylinders)
    - Interactive joint control via GUI sliders
    - IK (Inverse Kinematics) control with visual targets
    - Trajectory playback and recording
    - Camera controls and preset views

    Args:
        port: Port number for the viser server (default: 8080)
    """

    def __init__(self, port: int = 8080) -> None:
        self.server = viser.ViserServer(port=port)

    def enable_camera_controls(
        self, initial_position=None, 
        render_width=256, 
        render_height=256, 
        look_at_position=None, 
        initial_fov=45.0,
        camera_name="/camera_visualization",
    ):
        """Enable camera controls and recording for all connected clients.

        Args:
            initial_position: Initial camera position [x, y, z]. Default: [3.0, -3.0, 2.0]
            render_width: Width of rendered camera view. Default: 256
            render_height: Height of rendered camera view. Default: 256
            look_at_position: Initial look-at target [x, y, z]. Default: [0.0, 0.0, 0.0]
            initial_fov: Initial field of view in degrees. Default: 45.0
        """
        # Set default values if not provided
        if initial_position is None:
            initial_position = [3.0, -3.0, 2.0]
        if look_at_position is None:
            look_at_position = [0.0, 0.0, 0.0]

        # Create a single camera frustum for all clients
        camera_frustum = self.server.scene.add_camera_frustum(
            name="/camera_visualization",
            fov=np.radians(initial_fov),  # FOV in radians
            aspect=render_width / render_height,  # Aspect ratio based on render dimensions
            scale=0.3,
            color=(255, 255, 0),  # Yellow
            position=np.array(initial_position),
        )


        @self.server.on_client_connect
        def setup_client_camera(client):
            """Setup camera controls and recording for each connected client."""
            try:
                # Set main view camera using provided parameters
                client.camera.position = np.array([
                    initial_position[0] * 2,
                    initial_position[1] * 2,
                    initial_position[2] * 2,
                ])  # Offset main view
                client.camera.look_at = np.array(look_at_position)
                client.camera.fov = 45.0  # Keep main view FOV fixed

                # Create camera control GUI elements
                with client.gui.add_folder("Camera Controls"):
                    pos_x = client.gui.add_slider(
                        "Camera X", min=-10.0, max=10.0, step=0.2, initial_value=initial_position[0]
                    )
                    pos_y = client.gui.add_slider(
                        "Camera Y", min=-10.0, max=10.0, step=0.2, initial_value=initial_position[1]
                    )
                    pos_z = client.gui.add_slider(
                        "Camera Z", min=0.1, max=10.0, step=0.2, initial_value=initial_position[2]
                    )

                    # Add incremental camera rotation controls using buttons (compact layout)
                    with client.gui.add_folder("Camera Rotation"):
                        # Yaw controls (left/right turn) - compact pair
                        yaw_left_btn = client.gui.add_button("◄ Yaw")
                        yaw_right_btn = client.gui.add_button("► Yaw")

                        # Pitch controls (up/down tilt) - compact pair
                        pitch_up_btn = client.gui.add_button("▲ Pitch")
                        pitch_down_btn = client.gui.add_button("▼ Pitch")

                        # Roll controls (left/right roll) - compact pair
                        roll_left_btn = client.gui.add_button("↺ Roll")
                        roll_right_btn = client.gui.add_button("↻ Roll")

                    fov_slider = client.gui.add_slider(
                        "Camera FOV", min=20.0, max=90.0, step=1.0, initial_value=initial_fov
                    )

                    # Add button for one-time look-at center
                    lookat_center_btn = client.gui.add_button("Look At Center")

                    camera_info = client.gui.add_text("Camera Info", initial_value="Camera position and settings")
                    reset_btn = client.gui.add_button("Reset Camera")

                with client.gui.add_folder("Camera Presets"):
                    top_view_btn = client.gui.add_button("Top View")
                    side_view_btn = client.gui.add_button("Side View")
                    front_view_btn = client.gui.add_button("Front View")

                with client.gui.add_folder("Camera View"):
                    camera_view_info = client.gui.add_text("Camera View", initial_value="Camera view information")
                    screenshot_btn = client.gui.add_button("Take Camera Screenshot")

                    # Add recording controls
                    with client.gui.add_folder("Recording"):
                        recording_status = client.gui.add_text("Recording Status", initial_value="Not recording")
                        start_recording_btn = client.gui.add_button("Start Recording")
                        stop_recording_btn = client.gui.add_button("Stop Recording")
                        recording_fps = client.gui.add_slider("Recording FPS", min=5, max=30, step=1, initial_value=10)

                    # Add camera view image display in GUI
                    camera_image_gui = client.gui.add_image(
                        np.zeros((256, 256, 3), dtype=np.uint8),  # Initial empty image
                        label="Live Camera View",
                    )

                # Flag to prevent infinite loop when updating sliders programmatically
                updating_sliders = False

                # Recording state variables
                is_recording = False
                recording_frames = []
                recording_timer = None
                recording_start_time = None

                # Now define all functions that will be used
                def update_camera_view():
                    """Update the camera view display in GUI."""
                    try:
                        # Small delay to ensure scene updates are applied
                        time.sleep(0.01)

                        # Get camera settings from frustum
                        camera_pos = camera_frustum.position
                        camera_fov_radians = camera_frustum.fov
                        render_wxyz = camera_frustum.wxyz

                        # Calculate direction vector for debugging
                        look_at = np.array(look_at_position)
                        # direction = look_at - camera_pos
                        direction = camera_pos - look_at
                        direction = direction / np.linalg.norm(direction)

                        logger.info(f"Camera position: {camera_pos}")
                        logger.info(f"Look at: {look_at}")
                        logger.info(f"Direction: {direction}")
                        logger.info(f"Render wxyz: {render_wxyz}")
                        logger.info(f"FOV radians: {camera_fov_radians}")

                        # Try slightly offset position to avoid being exactly AT the frustum center
                        # Move slightly back along the negative direction vector
                        offset_distance = 0.01  # Small offset
                        offset_pos = camera_pos - direction * offset_distance

                        logger.info(f"Offset position: {offset_pos}")

                        # Temporarily hide camera frustum from render
                        # camera_frustum.visible = False

                        try:
                            # Render image from camera viewpoint (without seeing the frustum itself)
                            image = client.get_render(
                                height=render_height,
                                width=render_width,
                                wxyz=render_wxyz,
                                position=offset_pos,  # Use offset position
                                fov=camera_fov_radians,
                            )
                        finally:
                            # Always restore camera frustum visibility for 3D scene
                            camera_frustum.visible = True

                        logger.info(
                            f"Rendered image shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}"
                        )

                        # Check if image is completely black or has content
                        non_zero_pixels = np.count_nonzero(image)
                        total_pixels = image.size
                        logger.info(
                            f"Non-zero pixels: {non_zero_pixels}/{total_pixels} ({100 * non_zero_pixels / total_pixels:.1f}%)"
                        )

                        # Save original for debugging
                        original_image = image.copy()

                        # Update GUI image display
                        camera_image_gui.image = original_image
                    except Exception as e:
                        logger.error(f"Failed to update camera view: {e}")
                        import traceback

                        traceback.print_exc()

                self._camera_update_callback = update_camera_view

                def take_camera_screenshot(_):
                    """Take high-resolution camera screenshot."""
                    try:
                        import imageio

                        # Show feedback
                        camera_view_info.value = "Taking screenshot..."

                        # Get camera settings from frustum (same as GUI display)
                        camera_pos = camera_frustum.position
                        camera_fov_radians = camera_frustum.fov
                        render_wxyz = camera_frustum.wxyz

                        # Use same offset calculation as Live Camera View
                        look_at = np.array(look_at_position)
                        direction = look_at - camera_pos
                        direction = direction / np.linalg.norm(direction)
                        offset_distance = 0.01
                        offset_pos = camera_pos - direction * offset_distance

                        # Temporarily hide camera frustum from render
                        # camera_frustum.visible = False

                        try:
                            # Render high-resolution image (without seeing the frustum itself)
                            image = client.get_render(
                                height=render_height * 2,  # 2x resolution for screenshots
                                width=render_width * 2,
                                wxyz=render_wxyz,
                                position=offset_pos,  # Use offset position
                                fov=camera_fov_radians,
                            )
                        finally:
                            # Always restore camera frustum visibility for 3D scene
                            camera_frustum.visible = True

                            # Save image
                            filename = f"camera_view_{client.client_id}_{int(time.time())}.png"
                            imageio.imwrite(filename, image)

                        logger.info(f"Camera screenshot saved as {filename}")
                        camera_view_info.value = f"Screenshot saved: {filename}"

                    except Exception as e:
                        logger.error(f"Failed to take screenshot: {e}")
                        camera_view_info.value = f"Screenshot failed: {e}"

                def capture_recording_frame():
                    """Capture a frame for recording."""
                    nonlocal recording_frames
                    try:
                        # Get camera settings from frustum
                        camera_pos = camera_frustum.position
                        camera_fov_radians = camera_frustum.fov
                        render_wxyz = camera_frustum.wxyz

                        # Calculate offset position (same as screenshot)
                        look_at = np.array(look_at_position)
                        direction = look_at - camera_pos
                        direction = direction / np.linalg.norm(direction)
                        offset_distance = 0.01
                        offset_pos = camera_pos - direction * offset_distance

                        # Temporarily hide camera frustum from render
                        # camera_frustum.visible = False

                        try:
                            # Render frame for recording
                            image = client.get_render(
                                height=render_height,
                                width=render_width,
                                wxyz=render_wxyz,
                                position=offset_pos,
                                fov=camera_fov_radians,
                            )
                            recording_frames.append(image.copy())
                        finally:
                            # Always restore camera frustum visibility
                            camera_frustum.visible = True

                    except Exception as e:
                        logger.error(f"Failed to capture recording frame: {e}")

                def start_recording():
                    """Start recording camera view."""
                    nonlocal is_recording, recording_frames, recording_timer, recording_start_time

                    if is_recording:
                        return

                    is_recording = True
                    recording_frames = []
                    recording_start_time = time.time()

                    recording_status.value = "Recording..."
                    start_recording_btn.disabled = True
                    stop_recording_btn.disabled = False

                    # Start periodic frame capture
                    def capture_frames():
                        nonlocal recording_timer, recording_frames, recording_start_time
                        if is_recording:
                            capture_recording_frame()
                            frame_count = len(recording_frames)
                            elapsed_time = time.time() - recording_start_time
                            recording_status.value = f"Recording... {frame_count} frames ({elapsed_time:.1f}s)"

                            # Schedule next frame
                            recording_timer = threading.Timer(1.0 / recording_fps.value, capture_frames)
                            recording_timer.start()

                    capture_frames()
                    logger.info("Started camera recording")

                def stop_recording():
                    """Stop recording and save video."""
                    nonlocal is_recording, recording_timer, recording_frames

                    if not is_recording:
                        return

                    is_recording = False
                    if recording_timer:
                        recording_timer.cancel()
                        recording_timer = None

                    recording_status.value = "Saving video..."
                    start_recording_btn.disabled = False
                    stop_recording_btn.disabled = True

                    try:
                        import imageio

                        if len(recording_frames) > 0:
                            # Save video
                            filename = f"camera_recording_{client.client_id}_{int(time.time())}.mp4"
                            with imageio.get_writer(filename, fps=recording_fps.value) as writer:
                                for frame in recording_frames:
                                    writer.append_data(frame)

                            frame_count = len(recording_frames)
                            duration = frame_count / recording_fps.value
                            recording_status.value = f"Video saved: {filename} ({frame_count} frames, {duration:.1f}s)"
                            logger.info(f"Camera recording saved as {filename}")
                        else:
                            recording_status.value = "No frames recorded"
                            logger.warning("No frames were recorded")

                    except Exception as e:
                        logger.error(f"Failed to save recording: {e}")
                        recording_status.value = f"Save failed: {e}"
                    finally:
                        recording_frames = []

                # Store current camera rotation (starts as identity)
                current_camera_rotation = np.eye(3)

                def orthogonalize_matrix(R):
                    """Ensure rotation matrix remains orthogonal using Gram-Schmidt process."""
                    # Use SVD to get the closest orthogonal matrix
                    U, _, Vt = np.linalg.svd(R)
                    R_ortho = U @ Vt

                    # Ensure determinant is +1 (proper rotation, not reflection)
                    if np.linalg.det(R_ortho) < 0:
                        U[:, -1] *= -1
                        R_ortho = U @ Vt

                    return R_ortho

                def apply_incremental_rotation(axis, angle_degrees):
                    """Apply incremental rotation around specified axis."""
                    nonlocal current_camera_rotation

                    if abs(angle_degrees) < 1e-6:
                        return

                    angle_rad = np.radians(angle_degrees)
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

                    # Create rotation matrix for the specified axis
                    if axis == "roll":
                        R_delta = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
                    elif axis == "pitch":  # Around current X-axis
                        R_delta = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
                    else:
                        R_delta = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])

                    # Apply incremental rotation to current state
                    current_camera_rotation = current_camera_rotation @ R_delta

                    # Ensure the rotation matrix remains orthogonal to prevent deformation
                    current_camera_rotation = orthogonalize_matrix(current_camera_rotation)

                    # Update camera with new rotation
                    camera_frustum.wxyz = matrix_to_quaternion(current_camera_rotation)

                    # Update camera info display with new rotation
                    current_position = np.array([pos_x.value, pos_y.value, pos_z.value])
                    R = current_camera_rotation
                    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                    pitch = np.degrees(np.arcsin(-R[2, 0]))
                    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                    camera_info.value = f"Position: ({current_position[0]:.1f}, {current_position[1]:.1f}, {current_position[2]:.1f})\nRotation: R={roll:.1f}° P={pitch:.1f}° Y={yaw:.1f}°\nFOV: {fov_slider.value:.1f}°"

                    # Update camera view
                    import threading

                    threading.Thread(target=update_camera_view, daemon=True).start()

                def update_camera():
                    """Update camera position and settings."""
                    nonlocal updating_sliders

                    # Prevent infinite loop when we update sliders programmatically
                    if updating_sliders:
                        return

                    new_position = np.array([pos_x.value, pos_y.value, pos_z.value])
                    new_fov_degrees = fov_slider.value
                    new_fov_radians = np.radians(new_fov_degrees)

                    # Update camera frustum position and FOV
                    camera_frustum.position = new_position
                    camera_frustum.fov = new_fov_radians

                    # Extract Euler angles from current rotation matrix for display
                    R = current_camera_rotation
                    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                    pitch = np.degrees(np.arcsin(-R[2, 0]))
                    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

                    # Update camera info display
                    camera_info.value = f"Position: ({new_position[0]:.1f}, {new_position[1]:.1f}, {new_position[2]:.1f})\nRotation: R={roll:.1f}° P={pitch:.1f}° Y={yaw:.1f}°\nFOV: {new_fov_degrees:.1f}°"

                    # Update camera view
                    import threading

                    threading.Thread(target=update_camera_view, daemon=True).start()

                def look_at_center():
                    """Make camera look at specified target from current position."""
                    nonlocal current_camera_rotation

                    current_position = np.array([pos_x.value, pos_y.value, pos_z.value])
                    target = np.array(look_at_position)

                    # Calculate direction vector (from camera to target)
                    direction = target - current_position
                    distance = np.linalg.norm(direction)

                    # Check if we're too close to the center
                    if distance < 1e-6:
                        logger.warning("Camera is too close to the center, cannot look at itself")
                        return

                    # Normalize direction vector (this will be our +Z axis in camera space)
                    forward = direction / distance

                    # Choose world up vector (use Y-up convention, common in 3D graphics)
                    world_up = np.array([0.0, 0.0, 1.0])

                    # Calculate right vector (X axis in camera space)
                    right = np.cross(forward, world_up)
                    right_length = np.linalg.norm(right)

                    # Handle gimbal lock case (camera looking straight up or down)
                    if right_length < 1e-6:
                        # When looking straight up/down, use X-axis as reference
                        right = np.array([1.0, 0.0, 0.0])
                        if forward[1] < 0:  # looking down
                            right = np.array([-1.0, 0.0, 0.0])
                    else:
                        right = right / right_length

                    # Calculate up vector (Y axis in camera space) using cross product
                    # This ensures our coordinate system is orthogonal
                    up = np.cross(right, forward)
                    up = up / np.linalg.norm(up)  # Normalize to ensure orthogonality

                    # Construct rotation matrix (each column is an axis)
                    # OpenCV/Viser convention: +X right, +Y down, +Z forward
                    # So we need: [right, -up, forward] to match OpenCV convention
                    current_camera_rotation = np.column_stack([right, -up, forward])

                    # Ensure perfect orthogonality to prevent any deformation
                    current_camera_rotation = orthogonalize_matrix(current_camera_rotation)

                    # Verify orthogonality (debugging)
                    det = np.linalg.det(current_camera_rotation)
                    if abs(det - 1.0) > 1e-6:
                        logger.warning(f"Rotation matrix determinant is {det}, should be 1.0")

                    # Apply to camera frustum
                    camera_frustum.wxyz = matrix_to_quaternion(current_camera_rotation)

                    # Update camera info display with new rotation
                    R = current_camera_rotation
                    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                    pitch = np.degrees(np.arcsin(-R[2, 0]))
                    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                    camera_info.value = f"Position: ({current_position[0]:.1f}, {current_position[1]:.1f}, {current_position[2]:.1f})\nRotation: R={roll:.1f}° P={pitch:.1f}° Y={yaw:.1f}°\nFOV: {fov_slider.value:.1f}°"

                    # Update camera view
                    import threading

                    threading.Thread(target=update_camera_view, daemon=True).start()

                # Helper function for quaternion conversion
                def matrix_to_quaternion(R):
                    trace = R[0, 0] + R[1, 1] + R[2, 2]
                    if trace > 0:
                        s = np.sqrt(trace + 1.0) * 2
                        return np.array([
                            0.25 * s,
                            (R[2, 1] - R[1, 2]) / s,
                            (R[0, 2] - R[2, 0]) / s,
                            (R[1, 0] - R[0, 1]) / s,
                        ])
                    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                        return np.array([
                            (R[2, 1] - R[1, 2]) / s,
                            0.25 * s,
                            (R[0, 1] + R[1, 0]) / s,
                            (R[0, 2] + R[2, 0]) / s,
                        ])
                    elif R[1, 1] > R[2, 2]:
                        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                        return np.array([
                            (R[0, 2] - R[2, 0]) / s,
                            (R[0, 1] + R[1, 0]) / s,
                            0.25 * s,
                            (R[1, 2] + R[2, 1]) / s,
                        ])
                    else:
                        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                        return np.array([
                            (R[1, 0] - R[0, 1]) / s,
                            (R[0, 2] + R[2, 0]) / s,
                            (R[1, 2] + R[2, 1]) / s,
                            0.25 * s,
                        ])

                def reset_camera():
                    """Reset camera to initial position and orientation."""
                    nonlocal updating_sliders, current_camera_rotation
                    updating_sliders = True

                    # Reset position to initial values
                    pos_x.value = initial_position[0]
                    pos_y.value = initial_position[1]
                    pos_z.value = initial_position[2]

                    fov_slider.value = initial_fov

                    # Reset camera rotation to identity (default orientation)
                    current_camera_rotation = np.eye(3)
                    current_camera_rotation = orthogonalize_matrix(
                        current_camera_rotation
                    )  # Ensure perfect orthogonality
                    camera_frustum.wxyz = matrix_to_quaternion(current_camera_rotation)

                    updating_sliders = False
                    update_camera()

                def set_preset_view(position, rotation_matrix=None):
                    """Set camera to preset position and orientation."""
                    nonlocal updating_sliders, current_camera_rotation
                    updating_sliders = True

                    # Set position
                    pos_x.value = position[0]
                    pos_y.value = position[1]
                    pos_z.value = position[2]

                    # Set rotation if provided
                    if rotation_matrix is not None:
                        current_camera_rotation = rotation_matrix.copy()
                        current_camera_rotation = orthogonalize_matrix(
                            current_camera_rotation
                        )  # Ensure perfect orthogonality

                    updating_sliders = False

                    # Update camera with new preset
                    camera_frustum.position = np.array(position)
                    camera_frustum.wxyz = matrix_to_quaternion(current_camera_rotation)

                    # Update camera info display
                    R = current_camera_rotation
                    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                    pitch = np.degrees(np.arcsin(-R[2, 0]))
                    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                    camera_info.value = f"Position: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})\nRotation: R={roll:.1f}° P={pitch:.1f}° Y={yaw:.1f}°\nFOV: {fov_slider.value:.1f}°"

                    import threading

                    threading.Thread(target=update_camera_view, daemon=True).start()

                # Connect all GUI controls
                pos_x.on_update(lambda _: update_camera())
                pos_y.on_update(lambda _: update_camera())
                pos_z.on_update(lambda _: update_camera())
                fov_slider.on_update(lambda _: update_camera())
                lookat_center_btn.on_click(lambda _: look_at_center())
                reset_btn.on_click(lambda _: reset_camera())
                screenshot_btn.on_click(lambda _: take_camera_screenshot(_))

                # Connect recording buttons
                start_recording_btn.on_click(lambda _: start_recording())
                stop_recording_btn.on_click(lambda _: stop_recording())

                # Set initial recording button states
                stop_recording_btn.disabled = True

                # Connect rotation buttons (10 degree increments)
                step_angle = 10.0
                yaw_left_btn.on_click(lambda _: apply_incremental_rotation("yaw", -step_angle))
                yaw_right_btn.on_click(lambda _: apply_incremental_rotation("yaw", step_angle))
                pitch_up_btn.on_click(lambda _: apply_incremental_rotation("pitch", step_angle))
                pitch_down_btn.on_click(lambda _: apply_incremental_rotation("pitch", -step_angle))
                roll_left_btn.on_click(lambda _: apply_incremental_rotation("roll", -step_angle))
                roll_right_btn.on_click(lambda _: apply_incremental_rotation("roll", step_angle))

                # Connect preset buttons with appropriate rotation matrices
                @top_view_btn.on_click
                def top_view(_):
                    # Top view: looking straight down (rotate -90° around X-axis)
                    R_top = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                    set_preset_view([0.0, 0.0, 3.0], R_top)

                @side_view_btn.on_click
                def side_view(_):
                    # Side view: looking from +X towards origin (rotate 90° around Y-axis)
                    R_side = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
                    set_preset_view([3.0, 0.0, 1.0], R_side)

                @front_view_btn.on_click
                def front_view(_):
                    # Front view: looking from -Y towards origin (default orientation)
                    R_front = np.eye(3)
                    set_preset_view([0.0, -3.0, 1.0], R_front)

                # Initial setup
                update_camera()

                # If look_at_position is not the default origin, set initial look-at
                if look_at_position != [0.0, 0.0, 0.0]:
                    # Modify look_at_center function to use custom target
                    def initial_look_at():
                        nonlocal current_camera_rotation
                        current_position = np.array([pos_x.value, pos_y.value, pos_z.value])
                        target = np.array(look_at_position)

                        direction = target - current_position
                        distance = np.linalg.norm(direction)

                        if distance > 1e-6:
                            forward = direction / distance
                            world_up = np.array([0.0, 1.0, 0.0])
                            right = np.cross(forward, world_up)
                            right_length = np.linalg.norm(right)

                            if right_length < 1e-6:
                                right = np.array([1.0, 0.0, 0.0])
                                if forward[1] < 0:
                                    right = np.array([-1.0, 0.0, 0.0])
                            else:
                                right = right / right_length

                            up = np.cross(right, forward)
                            up = up / np.linalg.norm(up)

                            current_camera_rotation = np.column_stack([right, -up, forward])
                            current_camera_rotation = orthogonalize_matrix(current_camera_rotation)
                            camera_frustum.wxyz = matrix_to_quaternion(current_camera_rotation)

                    initial_look_at()

                # Delay camera view update to ensure frustum is properly set
                def delayed_camera_view_update():
                    time.sleep(0.01)  # Small delay
                    update_camera_view()

                import threading

                threading.Thread(target=delayed_camera_view_update, daemon=True).start()

                logger.info(f"Camera controls and recording enabled for client {client.client_id}")

            except Exception as e:
                logger.error(f"Failed to setup camera controls: {e}")
                import traceback

                traceback.print_exc()