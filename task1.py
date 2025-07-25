"""
task1: In this task, we will create a box and set its position and rotation randomly. The box will be created every 5 seconds, and the previous box will be deleted.
After the box is created, the stage will be fotographed and saved to files. Files include RGBA.  
"""

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

param_time = 300
param_iter = 0
param_numBoxes = 20
param_iter_soll = 50

stage = usd.get_context().get_stage()
layer = stage.GetRootLayer()

script_dir = "E:/VScode/VSworkspace/pyworkspace/isaacsimpy"
rgb_dir = f"{script_dir}/RGBFrame"
depth_dir = f"{script_dir}/depthFrame"

if not os.path.exists(rgb_dir):
    os.makedirs(rgb_dir)
    print(f"Created folder: {rgb_dir}")
else:
    print(f"Image folder already exists.")

if not os.path.exists(depth_dir):
    os.makedirs(depth_dir)
    print(f"Created folder: {depth_dir}")
else:
    print(f"Image folder already exists.")


def delete_box_copy():
    stage_box_copy = [stage.GetPrimAtPath(f"/World/warehouse_with_forklifts/SM_CardBoxC_Copy_{i}") for i in range(param_numBoxes)]
    for i, box_prim in enumerate(stage_box_copy):
        if box_prim.IsValid():
            print(f"Box {i}: {box_prim.GetPath()} is need to be deleted")
        else:
            print(f"Box {i}: Path not found")
    for box_prim in stage_box_copy:
        if box_prim.IsValid():
            stage.RemovePrim(box_prim.GetPath())
            print(f"Deleted {box_prim.GetPath()}")

delete_box_copy()
original_box_prim = stage.GetPrimAtPath("/World/warehouse_with_forklifts/SM_CardBoxC_01")
if not original_box_prim.IsValid():
    print("Error: Original box not found!")

camera = Camera(prim_path="/World/Camera_0", resolution=(1024, 512))

timeline = omni.timeline.get_timeline_interface()
timeline.play()

camera.initialize()

depth_annotator = AnnotatorRegistry.get_annotator("distance_to_camera")
depth_annotator.attach(camera.get_render_product_path())

async def my_task():
    global param_numBoxes, param_iter, param_time
    while True:
        # Wait for the next frame to be ready
        print(f"Timeline is running: {timeline.is_playing()}")
        print(f"time: {timeline.get_current_time()}")
        for i in range(param_numBoxes):
            trans_x = random.uniform(-5, 5)
            trans_y = random.uniform(-5, 5)
            trans_z = random.uniform(0.5, 3)

            rot_x = random.uniform(0, 360)
            rot_y = random.uniform(0, 360)
            rot_z = random.uniform(0, 360)
            new_box_path = f"/World/warehouse_with_forklifts/SM_CardBoxC_Copy_{i}"
            new_box_prim = stage.OverridePrim(new_box_path)
            new_box_prim.GetReferences().AddReference(assetPath="", primPath=original_box_prim.GetPath())
            if new_box_prim.IsValid():
                # 获取现有变换操作
                xform = UsdGeom.Xformable(new_box_prim)
                xform.ClearXformOpOrder()
                translate_op = xform.AddTranslateOp(opSuffix="")
                rotate_op = xform.AddRotateXYZOp(opSuffix="")
                translate_op.Set(Gf.Vec3d(trans_x, trans_y, trans_z))
                rotate_op.Set(Gf.Vec3f(rot_x, rot_y, rot_z)) 
                print(f"Created box_{i} at ({trans_x}, {trans_y}, {trans_z}) with rotation ({rot_x}, {rot_y}, {rot_z})")
        await asyncio.sleep(5)
        camera.add_motion_vectors_to_frame()
        await asyncio.sleep(1)

        # Get the RGBA image and save it
        rgb_image = camera.get_rgba()
        print(f" Get RGBA with Frame_{param_iter} is {rgb_image is not None}")
        bgr_image = cv2.cvtColor(rgb_image[..., :3], cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{rgb_dir}/rgbFrame_{param_iter}.png", bgr_image)

        # Get the depth image and save it
        depth_data = depth_annotator.get_data() 
        print(f" Get depth data with Frame_{param_iter} is {depth_data is not None}")
        depth_data_n = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_data_n = 255 - depth_data_n 
        depth_image = cv2.applyColorMap(depth_data_n, cv2.COLORMAP_JET)
        cv2.imwrite(f"{rgb_dir}/depthFrame_{param_iter}.png", depth_image)
        np.savetxt(f"{depth_dir}/depthData_{param_iter}.csv", depth_data, delimiter=' ')

        param_iter += 1
        await asyncio.sleep(1)
        delete_box_copy()
        await asyncio.sleep(1)
        # if timeline.get_current_time() >= param_time:
        #     print(f"Stopping timeline after {param_time} seconds.")
        #     break
        if param_iter >= param_iter_soll:
            print(f"Stopping timeline after {param_iter} iterations.")
            break

asyncio.ensure_future(my_task())
        

