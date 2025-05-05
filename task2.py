import omni.timeline
import asyncio
import omni.usd as usd
import random
import pxr.Gf as Gf
from pxr import UsdGeom
from isaacsim.sensors.camera import Camera
from omni.replicator.core import AnnotatorRegistry
import cv2
import os
import numpy as np

rgb_dir = "/RGBFrame"

if not os.path.exists(rgb_dir):
    os.makedirs(rgb_dir)
    print(f"Created folder: {rgb_dir}")
else:
    print(f"Image folder already exists.")

camera = Camera(prim_path="/World/Camera_0", resolution=(1024, 512))
camera.initialize()

async def my_task():
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    await asyncio.sleep(0.01)
    param_iter = 0
    while True:
        # Wait for the next frame to be ready
        camera.add_motion_vectors_to_frame()
        await asyncio.sleep(0.01)
        rgb_image = camera.get_rgba()
        print(f" Get RGBA with Frame_{param_iter} is {rgb_image is not None}")
        if rgb_image is not None:
            cv2.imwrite(f"{rgb_dir}/RGBFrame_{param_iter}.png", rgb_image)
            print(f"Saved RGB image as RGBFrame_{param_iter}.png")
        else:
            print("Failed to get RGB image.")
        param_iter += 1
        await asyncio.sleep(0.1)

asyncio.ensure_future(my_task())