from loguru import logger as log
# ========================================================================
# Setup Viser Visualization
# ========================================================================
from viser_utils import ViserVisualizer





# Initialize the viser server
visualizer = ViserVisualizer(port=8080)
# visualizer.add_grid()
# visualizer.add_frame("/world_frame")

# Enable camera controls
visualizer.enable_camera_controls(
    initial_position=[3.5, 0, 2.5],
    render_width=1280,
    render_height=960,
    look_at_position=[0, 0, 0.5],
    initial_fov=30,
)


log.info("Viser has been initialized, visit http://localhost:8080 to view the scene!")
import pdb; pdb.set_trace()
