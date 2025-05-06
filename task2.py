# task2: the creation of boxes is same as task1, but the camera will be moved to a random postion, where the viewpoint is the center of the scence. 
# After a few seconds, the scene will be photographed and saved to files. Files include RGB as png-pictures and depth as csv-file.
# As the same time, the labels of scence will also be saved as a json-file. The label include the Pose(position(xyz) and rotation(roll pitch yaw)) and instance semantics of the boxes.

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
from scipy.spatial.transform import Rotation as R

# Parameters
param_time = 300
param_iter = 0
param_numBoxes = 20
param_iter_soll = 50
target_point = np.array([8.0, np.random.uniform(-10, 10), np.random.uniform(1.5, 5)])
aimed_point = np.array([0, 0, 1.5])

# Functions
def bboxDict_to_transform(bbox_dict):
    """
    Input: bbox_dict: bounding_box_3d_anno.get_data()['data'][index_id]
    Output: center_world: the center of the box in world coordinates (x, y, z). shape: (3,)
            size_world: the size of the box in world coordinates (width, height, depth). shape: (3,)
            euler_angle: the rotation of the box in world coordinates (roll, pitch, yaw). shape: (3,)
    """
    corner = np.array([[bbox_dict[1], bbox_dict[2], bbox_dict[3]],
                       [bbox_dict[4], bbox_dict[5], bbox_dict[6]]])
    trans_mtx = bbox_dict[7]
    center_local = np.mean(corner, axis=0)
    center_local_1 = np.append(center_local, 1.0)
    trans_mtx_T = trans_mtx.reshape(4, 4).T
    center_world = trans_mtx_T @ center_local_1
    center_world = center_world[:3].tolist()
    rot_mtx = trans_mtx_T[:3, :3]
    ttr_mtx = trans_mtx_T[:3, 3]
    U, _, Vt = np.linalg.svd(rot_mtx)
    rot_mtx_pure = np.dot(U, Vt)
    r = R.from_matrix(rot_mtx_pure)
    euler_angle = r.as_euler('xyz', degrees=True)
    scale = np.array([np.linalg.norm(rot_mtx[:, 0]), np.linalg.norm(rot_mtx[:, 1]), np.linalg.norm(rot_mtx[:, 2])])
    size_local = (np.abs(corner[1] - corner[0])).tolist()
    size_world = scale * size_local
    return center_world, size_world, euler_angle

def camRotMtx_to_quaternion(R):
    """
    Input: R: rotation matrix. shape: (3, 3)
    Output: q: quaternion. shape: (4,)
    """
    K = np.zeros((4, 4))
    K[0, 0] = (1 + R[0,0] + R[1,1] + R[2,2]) / 4
    K[1, 1] = (1 + R[0,0] - R[1,1] - R[2,2]) / 4
    K[2, 2] = (1 - R[0,0] + R[1,1] - R[2,2]) / 4
    K[3, 3] = (1 - R[0,0] - R[1,1] + R[2,2]) / 4
    K[0,1] = K[1,0] = (R[1,2] - R[2,1]) / 4
    K[0,2] = K[2,0] = (R[2,0] - R[0,2]) / 4
    K[0,3] = K[3,0] = (R[0,1] - R[1,0]) / 4
    K[1,2] = K[2,1] = (R[0,1] + R[1,0]) / 4
    K[1,3] = K[3,1] = (R[2,0] + R[0,2]) / 4
    K[2,3] = K[3,2] = (R[1,2] + R[2,1]) / 4

    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)]
    if q[0] < 0:
        q = -q  
    return q

def camPosOri(target_point, aimed_point):
    """
    Input: target_point: the position of the camera. shape: (3,)
           aimed_point: the point that the camera is looking at. shape: (3,)
    Output: q: quaternion. shape: (4,)
    """
    x2 = (aimed_point - target_point) / np.linalg.norm(aimed_point - target_point)
    x1 = np.array([-1, 0, 0])
    y1 = np.array([0, -1, 0])
    z1 = np.array([0, 0, 1])
    y2 = np.array([- x1[1]/(np.sqrt(x1[0]**2 + x1[1]**2)), x1[0]/(np.sqrt(x1[0]**2 + x1[1]**2)), 0])
    z2 = np.cross(x2, y2)
    R_1to2 = np.linalg.inv(np.vstack((x2, y2, z2)).T) @ (np.vstack((x1, y1, z1)).T)
    R_0to1 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    R = R_1to2 @ R_0to1
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2,1], R[2,2])
    yaw = np.arctan2(R[1,0], R[0,0])
    q = camRotMtx_to_quaternion(R)
    return q, roll, pitch, yaw

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

# Directories for saving images and labels
script_dir = "E:/VScode/VSworkspace/pyworkspace/isaacsimpy"
rgb_dir = f"{script_dir}/RGBFrame"
depth_dir = f"{script_dir}/depthFrame"
label_dir = f"{script_dir}/labelFrame"

if not os.path.exists(rgb_dir):
    os.makedirs(rgb_dir)
    print(f"Created folder: {rgb_dir}")
else:
    print(f"Image folder already exists.")

if not os.path.exists(depth_dir):
    os.makedirs(depth_dir)
    print(f"Created folder: {depth_dir}")
else:
    print(f"Depth folder already exists.")

if not os.path.exists(label_dir):
    os.makedirs(label_dir)
    print(f"Created folder: {label_dir}")
else:
    print(f"Label folder already exists.")

# Start
stage = usd.get_context().get_stage()
layer = stage.GetRootLayer()

delete_box_copy()
original_box_prim = stage.GetPrimAtPath("/World/warehouse_with_forklifts/SM_CardBoxC_01")
if not original_box_prim.IsValid():
    print("Error: Original box not found!")

camera = Camera(prim_path="/World/Camera_0", resolution=(1024, 512))

async def my_task():
    global param_numBoxes, param_iter, param_time, target_point, aimed_point
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    await asyncio.sleep(0.1)
    camera.initialize()
    await asyncio.sleep(0.1)

    # Annotator Initialization
    depth_annotator = AnnotatorRegistry.get_annotator("distance_to_camera")
    depth_annotator.attach(camera.get_render_product_path())
    instanceSemantic_annotator = AnnotatorRegistry.get_annotator("instance_segmentation")
    instanceSemantic_annotator.attach(camera.get_render_product_path())
    bounding_box_3d_anno = AnnotatorRegistry.get_annotator("bounding_box_3d")
    bounding_box_3d_anno.attach(camera.get_render_product_path())

    # Main Loop
    while True:
        await asyncio.sleep(0.1)
        q, camRoll, camPitch, camYaw = camPosOri(target_point, aimed_point)
        camera.set_world_pose(
            position=target_point,
            orientation=q,  
        ) 
        await asyncio.sleep(0.1)
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