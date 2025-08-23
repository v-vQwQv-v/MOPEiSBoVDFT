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
print(f"bounding_box_3d_data: {bounding_box_3d_data}")
print(f"bounding_box_3d_data['info']['idToLabels']: {bounding_box_3d_data['info']['idToLabels']}")
# 第五段
bounding_box_3d_dd = bounding_box_3d_data['data']
bbox_ids = bounding_box_3d_data['info']['bboxIds'] 
prim_paths = bounding_box_3d_data['info']['primPaths']
id_to_labels = bounding_box_3d_data['info']['idToLabels']

# 不需要打印信息
print(f"bounding_box_3d_dd: {bounding_box_3d_dd}")
print(f"bounding_box_3d_dd: {bounding_box_3d_dd.shape}")
print(f"bounding_box_3d_info: {bounding_box_3d_data['info']}")
print(f"prim_paths: {len(prim_paths)}")
print(f"bbox_ids: {bbox_ids}")


box_prim_path = f"/World/warehouse_with_forklifts/SM_CardBoxC_Copy_4/SM_CardBoxC_01"

try:
    index_id = prim_paths.index(box_prim_path)
    bbox_id = bbox_ids[index_id]
    print(f"prim_path index is {index_id}, the bbox_id is {bbox_id}")
    bbox_dict = bounding_box_3d_dd[index_id]
    print(f"bbox_dict: {bbox_dict}")
    corner = np.array([[bounding_box_3d_dd[index_id][1], bounding_box_3d_dd[index_id][2], bounding_box_3d_dd[index_id][3]],
            [bounding_box_3d_dd[index_id][4], bounding_box_3d_dd[index_id][5], bounding_box_3d_dd[index_id][6]]])
    print(f"corner: {corner}")
    trans_mtx = bounding_box_3d_dd[index_id][7]
except ValueError:
    print(f"no prim_path: {box_prim_path}")

# 第六段: 验证
import numpy as np
from pxr import Usd, UsdGeom, Gf
import omni.usd

# 计算变换
center_local = np.mean(corner, axis=0)
center_local_1 = np.append(center_local, 1.0)
trans_mtx_T = trans_mtx.reshape(4, 4).T
center_world = trans_mtx_T @ center_local_1
center_world = center_world[:3].tolist()
print(f"center_world: {center_world}")

rot_mtx = trans_mtx_T[:3, :3]
ttr_mtx = trans_mtx_T[:3, 3]
print(f"rot_mtx: {rot_mtx}, ttr_mtx: {ttr_mtx}")
U, _, Vt = np.linalg.svd(rot_mtx)
rot_mtx_pure = np.dot(U, Vt)
r = R.from_matrix(rot_mtx_pure)
print(f"r: {r}")
euler_angle = r.as_euler('xyz', degrees=True)
print(f"euler_angle: {euler_angle}")



scale = np.array([np.linalg.norm(rot_mtx[:, 0]), np.linalg.norm(rot_mtx[:, 1]), np.linalg.norm(rot_mtx[:, 2])])
print(f"scale: {scale}")
size_local = (np.abs(corner[1] - corner[0])).tolist()
size_world = scale * size_local
print(f"size_local: {size_local}, size_world: {size_world}")


# 可视化
stage = omni.usd.get_context().get_stage()

bbox_prim = stage.DefinePrim("/World/OBB_Visual", "Cube")

bbox_geom = UsdGeom.Cube(bbox_prim)
bbox_geom.CreateSizeAttr(1.0)

color_attr = bbox_geom.CreateDisplayColorAttr()
color_attr.Set([(1.0, 0.0, 0.0)])  # 红色

xform = UsdGeom.Xform(bbox_prim)

xform.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*center_world))
xform.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(size_world)))
xform.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ, UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*euler_angle))




