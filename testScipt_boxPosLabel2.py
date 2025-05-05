# 利用render调用BoundingBox3D
import asyncio
from isaacsim.sensors.camera import Camera
from omni.replicator.core import AnnotatorRegistry
import omni.timeline
import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

script_dir = "E:/VScode/VSworkspace/pyworkspace/isaacsimpy"
camera = Camera(prim_path="/World/Camera_0", resolution=(1024, 512))
param_numBoxes = 20

# 第二段
camera.initialize()

timeline = omni.timeline.get_timeline_interface()
timeline.play()

# 第三段
camera.add_motion_vectors_to_frame()
print(f"render_product_path: {camera.get_render_product_path()}")

# 第四段
bounding_box_3d_anno = AnnotatorRegistry.get_annotator("bounding_box_3d")
bounding_box_3d_anno.attach(camera.get_render_product_path())

# 第五段
bounding_box_3d_data = bounding_box_3d_anno.get_data()

def compute_pose_from_bbox(corner, trans_mtx, euler_order='xyz', degrees=True):
    """
    参数：
        corner: np.array with shape (2, 3), [min_xyz, max_xyz]
        trans_mtx: np.array with shape (4, 4), 变换矩阵
        euler_order: 欧拉角顺序(默认为 'xyz' -> Roll, Pitch, Yaw)
        degrees: 是否返回角度(True)或弧度(False)

    返回：
        pos: np.array([x, y, z])，世界坐标下中心位置
        rot_angle: np.array([roll, pitch, yaw])，欧拉角(角度或弧度)
    """

    # 计算局部中心点
    local_center = (corner[0] + corner[1]) / 2.0
    local_center_h = np.append(local_center, 1.0)  # 齐次坐标 [x, y, z, 1]

    # 应用变换，得到世界坐标中心
    world_center_h = trans_mtx @ local_center_h
    pos = world_center_h[:3]

    # 提取旋转矩阵（前 3x3）
    rotation_matrix = trans_mtx[:3, :3]

    # 计算欧拉角
    r = R.from_matrix(rotation_matrix)
    rot_angle = r.as_euler(euler_order, degrees=degrees)

    return pos, rot_angle

# 第五段
bounding_box_3d_dd = bounding_box_3d_data['data']
bbox_ids = bounding_box_3d_data['info']['bboxIds']
prim_paths = bounding_box_3d_data['info']['primPaths']
id_to_labels = bounding_box_3d_data['info']['idToLabels']

# 不需要打印信息
print(f"bounding_box_3d_dd: {bounding_box_3d_dd.shape}")
print(f"bounding_box_3d_info: {bounding_box_3d_data['info']}")
print(f"prim_paths: {len(prim_paths)}")


box_prim_path = f"/World/warehouse_with_forklifts/SM_CardBoxC_Copy_14/SM_CardBoxC_01"

try:
    index_id = prim_paths.index(box_prim_path)
    bbox_id = bbox_ids[index_id]
    print(f"prim_path index is {index_id}, the bbox_id is {bbox_id}")
    bbox_dict = bounding_box_3d_dd[index_id]
    print(f"bbox_dict: {bbox_dict}")
    corner = np.array([[bounding_box_3d_dd[index_id][1], bounding_box_3d_dd[index_id][2], bounding_box_3d_dd[index_id][5]],
                        [bounding_box_3d_dd[index_id][3], bounding_box_3d_dd[index_id][4], bounding_box_3d_dd[index_id][6]]])
    print(f"corner: {corner}")
    size = np.abs(corner[1] - corner[0])
    print(f"size: {size}")
    trans_mtx = bounding_box_3d_dd[index_id][7]
    pos, rot_angle = compute_pose_from_bbox(corner, trans_mtx)
    print(f"pos: {pos}, euler_angle: {rot_angle}")
except ValueError:
    print(f"no prim_path: {box_prim_path}")

# 第六段: 验证
import omni.usd as usd
from pxr import Gf, UsdGeom
stage = usd.get_context().get_stage()
bbox_prim = stage.DefinePrim("/World/AABB_Visual", "Cube")
bbox_prim.GetAttribute("size").Set(1.0)
bbox_prim.GetAttribute("primvars:displayColor").Set([(1, 0, 0)])  

scale_xform = UsdGeom.Xform(bbox_prim)
size_vec = [float(x) / 2.0 for x in size]  # 转成 float64 并除以 2
scale_xform.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*size_vec))

matrix = Gf.Matrix4d(trans_mtx.astype(np.float64).reshape(4, 4).T)

# 添加或设置 transform 操作
xform_op = scale_xform.AddXformOp(
    UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble
)
xform_op.Set(matrix)







