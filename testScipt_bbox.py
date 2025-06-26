# 边界框
import asyncio
from isaacsim.sensors.camera import Camera
from omni.replicator.core import AnnotatorRegistry
import omni.timeline
import cv2
import numpy as np
import json

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
instanceSemantic_annotator = AnnotatorRegistry.get_annotator("instance_segmentation")
instanceSemantic_annotator.attach(camera.get_render_product_path())

