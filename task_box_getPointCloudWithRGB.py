"""
task_box_getPointCloudWithRGB: 
    This script captures point cloud data from a scene in NVIDIA Isaac Sim.
    The object in the scene is a strandard box.
"""

import omni.replicator.core as rep
import omni.syntheticdata as sd
import numpy as np
import asyncio
from pxr import UsdGeom, Gf, Usd, UsdPhysics, PhysxSchema
import omni.usd as usd
from isaacsim.sensors.camera import Camera
import os
import omni.timeline
from omni.replicator.core import AnnotatorRegistry
from scipy.spatial.transform import Rotation as R
import omni.kit.viewport.utility as viewport_utility
import os

"""Parameters"""
objectName = "SM_CardBoxC_01"
objectprimPath = f"/Root/{objectName}"
camPosScale = 4.0 # meter, the carmeras will be created in every corner of {camPosScale} x {camPosScale} x {camPosScale} cube
root_dir = "E:/VScode/VSworkspace/pyworkspace/isaacsimpy"

primPath = f"/Root/{objectName}"

aimed_point = np.array([0, 0, 0])  # The point the camera is aimed at
list_target_point = np.array([[camPosScale, camPosScale, camPosScale],
                         [-camPosScale, camPosScale, camPosScale],
                         [camPosScale, -camPosScale, camPosScale],
                         [-camPosScale, -camPosScale, camPosScale],
                         [camPosScale, camPosScale, -camPosScale],
                         [-camPosScale, camPosScale, -camPosScale],
                         [camPosScale, -camPosScale, -camPosScale],
                         [-camPosScale, -camPosScale, -camPosScale]])

def quaternion2rpy_XYZ(q):
    """
    将四元数 (w, x, y, z) 转换为欧拉角 (roll, pitch, yaw)，按 XYZ 顺序。
    返回值单位为弧度。
    """
    w, x, y, z = q

    # 计算 roll (绕 X 轴旋转)
    sinr = 2.0 * (w * x - y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)

    # 计算 pitch (绕 Y 轴旋转)
    sinp = 2.0 * (w * y + z * x)
    sinp = np.clip(sinp, -1.0, 1.0)  # 保证数值稳定
    pitch = np.arcsin(sinp)

    # 计算 yaw (绕 Z 轴旋转)
    siny = 2.0 * (w * z - x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)

    return np.array([roll, pitch, yaw])

def rotMtx2quaternion(R):
    """
    将 3x3 旋转矩阵转换为四元数 (w, x, y, z)。

    参数:
        R: 3x3 旋转矩阵 (numpy.ndarray)

    返回:
        四元数 (w, x, y, z)
    """
    if R.shape != (3, 3):
        raise ValueError(f"输入必须是 3x3 的旋转矩阵, 但得到的是形状 {R.shape} 的数组。")

    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S = 4 * w
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * x
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * y
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * z
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return np.array([w, x, y, z])

def quaternion2rotMtx(q):
    """
    Input: q: quaternion. shape: (4,)
    Output: R: rotation matrix. shape: (3, 3)
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2,   2*x*y - 2*w*z,          2*x*z + 2*w*y],
        [2*x*y + 2*w*z,         1 - 2*x**2 - 2*z**2,    2*y*z - 2*w*x],
        [2*x*z - 2*w*y,         2*y*z + 2*w*x,          1 - 2*x**2 - 2*y**2]
    ])
    return R

def camPosOri(target_point, aimed_point, R_w0toc0=np.eye(3)):
    """
    Input:
        target_point: shape (3,)
        aimed_point: shape (3,)
        
    Output:
        q: quaternion (w,x,y,z)
    """

    """Aim the camera at the target point """
    z_1 = -(aimed_point - target_point) / np.linalg.norm(aimed_point - target_point)

    """The Original Coordinate System of the Camera"""
    x_0 = np.array([1, 0, 0])
    y_0 = np.array([0, 1, 0])
    z_0 = np.array([0, 0, 1])

    """
    The New Coordinate System of the Camera: 
        y_1 should vertical to z_1 and z_0
        x_1 should vertical to y_1 and z_1
    """
    x_1 = np.cross(z_0, z_1)/ np.linalg.norm(np.cross(z_0, z_1))
    y_1 = np.cross(z_1, x_1) / np.linalg.norm(np.cross(z_1, x_1))

    """
    Rotation Matrix from the Original Coordinate System to the New Coordinate System:
        R @ C_0 = C_1
        R = C_1 @ C_0^(-1)
    """
    C_0 = np.vstack((x_0, y_0, z_0)).T  # Original Coordinate System
    C_1 = np.vstack((x_1, y_1, z_1)).T  # New Coordinate System
    R_w0tow1 = C_1 @ np.linalg.inv(C_0)  # Rotation Matrix from the Original to the New Coordinate System
    R_w0toc1 = R_w0tow1 @ R_w0toc0# Rotation Matrix from the Original Camera Coordinate System to the New Camera Coordinate System
    q = rotMtx2quaternion(np.array(R_w0toc1))  # Convert Rotation Matrix to Quaternion
    # print(f"R_w0toc0: {R_w0toc0}, R_c0toc1: {R_c0toc1}, q: {q}")
    return q

def removeNearDuplicatePoints(list_xyzrgb, threshold=1e-3):
    """
    从列表中移除位置（xyz）近似重复的点，仅保留第一个出现的点。
    
    Args:
        list_xyzrgb (list): 每个元素为 [x, y, z, r, g, b]
        threshold (float): xyz 距离判定的阈值，单位与 xyz 一致

    Returns:
        list: 清洗后的列表
    """
    kept_points = []
    xyz_set = []

    for pt in list_xyzrgb:
        xyz = np.array(pt[:3])
        is_duplicate = False

        for ref_xyz in xyz_set:
            if np.linalg.norm(xyz - ref_xyz) < threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_points.append(pt)
            xyz_set.append(xyz)

    return np.array(kept_points).reshape(-1, 6)  # Reshape to ensure the output is a 2D array with 6 columns (x, y, z, r, g, b)

async def my_task():
    """ Set the center of object to [0, 0, 0]"""
    stage = usd.get_context().get_stage()
    objectprim = stage.GetPrimAtPath(objectprimPath)
    geom = UsdGeom.Boundable(objectprim)
    bbox = geom.ComputeLocalBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
    min_point = bbox.GetRange().GetMin()
    max_point = bbox.GetRange().GetMax()
    center_point = (min_point + max_point) / 2.0
    print(f"min_point: {min_point}, max_point: {max_point}, center_point: {center_point}")
    xform = UsdGeom.Xformable(objectprim)
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(-center_point[0], -center_point[1], -center_point[2]))
    rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(objectprim)
    rigid_api.CreateDisableGravityAttr().Set(True)

    # cam = rep.create.camera(position=(camPosScale, camPosScale, camPosScale))
    target_point = np.array([camPosScale, camPosScale, camPosScale])
    camera_prim_path = "/Root/MyCamera"
    if not stage.GetPrimAtPath(camera_prim_path):
        stage.DefinePrim(camera_prim_path, "Camera")
    camera = Camera(prim_path=camera_prim_path, resolution=(1024, 512))
    camera.set_world_pose(orientation=[-0.5, 0.5, -0.5, -0.5])

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    await asyncio.sleep(2)
    viewport_name = viewport_utility.get_viewport_from_window_name("Viewport")
    viewport_name.camera_path = camera_prim_path
    await asyncio.sleep(2)
    camera.initialize()
    await asyncio.sleep(0.1)
    pcd_annotator = AnnotatorRegistry.get_annotator("pointcloud")
    await asyncio.sleep(0.1)
    pcd_annotator.attach(camera.get_render_product_path())
    await asyncio.sleep(0.1)

    list_xyzrgb = []
    for target_point in list_target_point:
        q = camPosOri(target_point, aimed_point, R_w0toc0=quaternion2rotMtx([-0.5, 0.5, -0.5, -0.5]))
        print(f"q: {q}")
        camera.set_world_pose(position=target_point, orientation=q)
        await asyncio.sleep(2)
        camera.add_motion_vectors_to_frame()
        
        """get the point cloud data"""
        pcd_data = pcd_annotator.get_data()
        xyzrgb = np.hstack((pcd_data['data'], pcd_data['pointRgb'][:, :3]))
        list_xyzrgb.append(xyzrgb)
        await asyncio.sleep(2)
    
    pcd = removeNearDuplicatePoints(list_xyzrgb)
    np.savetxt(f"{root_dir}/{objectName}_pcd.csv", pcd, delimiter=' ')
    



asyncio.ensure_future(my_task())

"""test the point cloud data"""
