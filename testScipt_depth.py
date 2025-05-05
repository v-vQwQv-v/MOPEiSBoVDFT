# 深度图需要分部执行
import asyncio
from isaacsim.sensors.camera import Camera
from omni.replicator.core import AnnotatorRegistry
import omni.timeline
import cv2

script_dir = "E:/VScode/VSworkspace/pyworkspace/isaacsimpy"
camera = Camera(prim_path="/World/Camera_0", resolution=(1024, 512))

# 第二段
camera.initialize()

timeline = omni.timeline.get_timeline_interface()
timeline.play()

# 第三段
camera.add_motion_vectors_to_frame()
print(f"render_product_path: {camera.get_render_product_path()}")

# 第四段
depth_annotator = AnnotatorRegistry.get_annotator("distance_to_camera")
depth_annotator.attach(camera.get_render_product_path())

# 第五段
depth_data = depth_annotator.get_data() 

print(f"depth_data: {depth_data}")
print(f"depth_data shape: {depth_data.shape}")

depth_data_n = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
depth_data_n = 255 - depth_data_n 
depth_image = cv2.applyColorMap(depth_data_n, cv2.COLORMAP_JET)

cv2.imwrite(f"{script_dir}/test_depth.png", depth_image)
