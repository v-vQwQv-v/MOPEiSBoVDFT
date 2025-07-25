from isaacsim.sensors.camera import Camera
import omni.timeline
import cv2
import omni.usd
from pxr import UsdGeom

# # 1
# camera = Camera(prim_path="/World/Camera_0", resolution=(1024, 512))

# # 2
# camera.initialize()

# # 3
# timeline = omni.timeline.get_timeline_interface()
# timeline.play()

# # 4
# camera.add_motion_vectors_to_frame()

# 5
stage = omni.usd.get_context().get_stage()
camera_prim = UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera_0"))
projection = camera_prim.GetProjectionAttr().Get()
print("Projection type:", projection)