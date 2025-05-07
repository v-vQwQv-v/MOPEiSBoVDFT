import numpy as np
from omni.isaac.sensor import Camera
from pxr import Gf


camera = Camera(prim_path="/World/Camera_0")

target_point = np.array([8.0, np.random.uniform(-10, 10), np.random.uniform(1.5, 5)]) 
aimed_point = np.array([0, 0, 1.5])

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
print(f"pitch: {np.rad2deg(pitch)}, roll: {np.rad2deg(roll)}, yaw: {np.rad2deg(yaw)}")

def rotation_matrix_to_quaternion(R):
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

    # 求最大特征值对应的特征向量
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)]
    if q[0] < 0:
        q = -q  # 保证w>0
    return q  # 返回w, x, y, z

q = rotation_matrix_to_quaternion(R)
print(f"q: {q}")

camera.set_world_pose(
    position=target_point,
    orientation=q,  
) 



camera.set_world_pose(
    position=[8.0, 0.0, 1.5],
    orientation=[0, 0, 0, 1],  
) 

# 获取相机位姿、焦距等信息
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation as R
import omni.usd as usd

def get_prim_pose(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim '{prim_path}' not found.")

    xform = UsdGeom.Xform(prim)
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # 位置
    translation = matrix.ExtractTranslation()

    # rotation matrix -> quaternion
    rotation_matrix = matrix.ExtractRotationMatrix()
    rot_np = np.array(rotation_matrix.GetTranspose())  # 注意转置
    quat = R.from_matrix(rot_np).as_quat()  # [x, y, z, w] format

    return {
        "position": (translation[0], translation[1], translation[2]),
        "quaternion_xyzw": quat.tolist(),  # [x, y, z, w]
    }


stage = usd.get_context().get_stage()
pose = get_prim_pose(stage, "/World/Camera_0")
print(pose)



horizontal_aperture = camera.get_horizontal_aperture()  # 单位：毫米      # 单位：毫米
focal_length = camera.get_focal_length() 

print(f"horizontal_aperture: {horizontal_aperture}, vertical_aperture: {horizontal_aperture * (512 / 1024)}, focal_length: {focal_length}")