"""
# task3.py
This Script is designed to create a dataset for indoor-logistic scene with multi-objects.
It includes the following features:
    1. SM-Boxes
    2. Forklifts 
    3. Pallets with boxes
    4. Blue boxes
    5. RackLarge
"""

import omni.timeline
import asyncio
import omni.usd as usd
import random
import pxr.Gf as Gf
from pxr import Usd, UsdGeom, Sdf
from isaacsim.sensors.camera import Camera
from omni.replicator.core import AnnotatorRegistry
import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import json

"""0. Parameters"""
"""0.0 Path Management"""
taskNum = '3'
path_dir_root = "E:/VScode/VSworkspace/pyworkspace/isaacsimpy"
path_dir_script = f"{path_dir_root}/dataset_task_{taskNum}"
path_dir_rgb = f"{path_dir_script}/RGBFrame"
path_dir_depth = f"{path_dir_script}/depthFrame"
path_dir_label = f"{path_dir_script}/labelFrame"

"""0.1 Adjustable Parameters"""
"""0.1.1 Scene Parameters"""
aParam_max_rdLarge = 6
aParam_max_pwbh = 6
aParam_max_forklift = 2
aParam_max_dumper = 2
aParam_max_rdLarge_smBox = [30, 20, 10]
aParam_max_pwbh_smBox = [20, 10, 5]
aParam_max_rdLarge_container = 10
aParam_distriBoxWeightAlpha = 2.0 # Distribution of box weights

"""0.1.2 Time Parameters"""
aParam_max_time = 10000
aParam_iter = 0
aParam_max_iter = 300

"""0.1.3 Camera Parameters"""
aParam_aimedPoint = np.array([0, 0, 1.5])
aParam_imgWidth = 1024
aParam_imgHeight = 512

"""0.2 Unadjustable Parameters"""
uPath_camera = "/World/Camera_0"
"""0.2.1 Scene Parameters"""
uParam_transScale = 100.0 # Translation scale for objects in cm to meters
uParam_smBoxSizeList = [[0.26, 0.35, 0.16], [0.52, 0.52, 0.26], [0.52, 0.73, 0.52]] # small, medium, large, used for rdLarges and pwbhs
uParam_supportThreshold = 0.8 # Support threshold, in scale
uParam_tolerance = 0.05 # Tolerance for translation, in scale
uParam_sceneSize = [20.0, 24.0] # Size of the scene [L, W]

"""0.2.2 RackLarge"""
uParam_rdLarge_Space =  [1.84, 3.7, 1.84] # Space of RackLarge [L, W, H]
uParam_rdLarge_Height_1 = 0.5 # Height of the first layer, in meters
uParam_rdLarge_Height_2 = 2.38 # Height of the second layer, in meters
uParam_rdLarge_mtxTrans = uParam_transScale * np.array([[1, 0,  0, -uParam_rdLarge_Space[0]/2 + uParam_tolerance],
                                                        [0, 1,  0, -uParam_rdLarge_Space[1]/2 + uParam_tolerance],
                                                        [0, 0,  1, uParam_rdLarge_Height_1],
                                                        [0, 0,  0, 1]], dtype=np.float32)
uParam_rdLarge_yaw0 = 90
uPath_rdLarge = "/World/RackLarge_A1"
uPath_rdLarge_smBoxList_original = [
    "/World/RackLarge_A1/Cardbox_D2/Cardbox_D2",
    "/World/RackLarge_A1/Cardbox_C3/Cardbox_C3",
    "/World/RackLarge_A1/Cardbox_A3/Cardbox_A3"
]
uPath_rdLarge_container = "/World/RackLarge_A1/Container_B09_40x30x22cm_PR_V_NVD_01"

"""0.2.3 Pallet"""
uParam_pwbh_Space = [1.70, 1.90, 1.80] # Space of Pallet [L, W, H]
uParam_pwbh_Height = 0.21 # Height of the first layer, in meters
uParam_pwbh_mtxTrans = uParam_transScale * np.array([[1, 0,  0, -uParam_pwbh_Space[0]/2 + uParam_tolerance],
                                                     [0, 1,  0, -uParam_pwbh_Space[1]/2 + uParam_tolerance],
                                                     [0, 0,  1, uParam_pwbh_Height],
                                                     [0, 0,  0, 1]], dtype=np.float32)
uParam_pwbh_yaw0 = 0
uPath_pwbh = "/World/WarehousePile_A6"
uPath_pwbh_smBoxList_original = [
    "/World/WarehousePile_A6/Cardbox_D2",
    "/World/WarehousePile_A6/Cardbox_C1",
    "/World/WarehousePile_A6/Cardbox_A3"
]
"""0.2.4 Forklift"""
uPath_forklift = "/World/warehouse_with_forklifts/Forklift"

"""0.2.5 Dumper"""
uParam_dumper_offsetXYZ = [-0.8, 0, 0.45]
uParam_dumper_transScale = 1000.0
uParam_dumper_transScale_T = 1/uParam_dumper_transScale
uPath_dumper = "/World/_9684481"

"""1. Functions"""
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
    return center_world, size_world, euler_angle.tolist()

def rotMtx2quaternion(R):
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
    # pitch = np.arcsin(-R[2, 0])
    # roll = np.arctan2(R[2,1], R[2,2])
    # yaw = np.arctan2(R[1,0], R[0,0])
    q = rotMtx2quaternion(R)
    return q

def deleteCopy(stage):
    """
    Delete the copy of the original bounding box.
    """
    prefixesDelete = [
        "/World/RackLarge_A1_Copy_",
        "/World/WarehousePile_A6_Copy_",
        "/World/Dumper_Copy_",
    ]
    for prim in stage.GetPrimAtPath("/World").GetChildren():
        path = str(prim.GetPath())
        for pref in prefixesDelete:
            if path.startswith(pref):
                stage.RemovePrim(prim.GetPath())
                print(f"Deleted prim: {path}")
                break
    for prim in stage.GetPrimAtPath("/World/warehouse_with_forklifts").GetChildren():
        path = str(prim.GetPath())
        if path.startswith("/World/warehouse_with_forklifts/ForkLift_Copy_"):
            stage.RemovePrim(prim.GetPath())
            print(f"Deleted prim: {path}")
            break

def get_obj_pose(stage, prim_path):
    """
    Input: stage: the stage of the scene.
           prim_path: the path of the object. e.g. "/World/warehouse_with_forklifts/SM_CardBoxC_Copy_0"
    Output: obj_pose: the pose of the object. [x, y, z, qx, qy, qz, qw]
    """
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

    return [translation[0], translation[1], translation[2], quat[0], quat[1], quat[2], quat[3]]  # [x, y, z], [x, y, z, w]

def serialize_label_data(label_dict, filename):
    def format_array_2d(arr, indent_level=2):
        indent = ' ' * (indent_level * 4)
        lines = []
        for row in arr:
            row_str = ', '.join(str(int(v)) if isinstance(v, (int, np.integer)) else f"{v:.6g}" for v in row)
            lines.append(indent + "[" + row_str + "]")
        return "[\n" + ",\n".join(lines) + "\n" + (' ' * 4 * (indent_level - 1)) + "]"

    formatted_json = "{\n"

    # 普通字段直接写
    for key in ["id", "camPose", "camParam"]:
        formatted_json += f'    "{key}": {json.dumps(label_dict[key], ensure_ascii=False)},\n'

    # instanceSemantics 保留原 shape 排列
    formatted_json += f'    "instanceSemantics": {format_array_2d(label_dict["instanceSemantics"], indent_level=2)},\n'

    # objPose 也可按原 shape（每行为一个 box）
    formatted_json += f'    "objPose": {format_array_2d(label_dict["objPose"], indent_level=2)}\n'

    formatted_json += "}"

    # 写入文件
    with open(filename, "w", encoding="utf-8") as f:
        f.write(formatted_json)

def pack_boxes_weighted_random(spaceSize, listBoxTypeSize, listBoxQuantity, 
                               support_threshold=0.5, alpha=2.0, max_fail=50,
                               w_x=1, w_y=1, w_z=0.5, rdYaws=False):
    Ls, Ws, Hs = spaceSize

    # 初始化物品池
    box_pool = []
    for i,(dims,k) in enumerate(zip(listBoxTypeSize, listBoxQuantity)):
        for _ in range(k):
            box_pool.append((i,dims))
    
    placed_boxes = []
    occupied = []
    free_points = {(0,0,0,w_z)}  # 初始可用点
    fail_count = 0
    EPS = 1e-6

    def has_support(x,y,z,L,W):
        if z==0:
            return True
        support_area=0
        for bx,by,bz,bL,bW,bH in occupied:
            if abs(z-(bz+bH))<EPS:
                overlap_x=max(0,min(x+L,bx+bL)-max(x,bx))
                overlap_y=max(0,min(y+W,by+bW)-max(y,by))
                support_area+=overlap_x*overlap_y
        return support_area>=support_threshold*L*W

    def can_place(x,y,z,L,W,H):
        if x<0 or y<0 or z<0 or x+L>Ls or y+W>Ws or z+H>Hs:
            return False
        if not has_support(x,y,z,L,W):
            return False
        for bx,by,bz,bL,bW,bH in occupied:
            if not (x+L<=bx+EPS or bx+bL<=x+EPS or
                    y+W<=by+EPS or by+bW<=y+EPS or
                    z+H<=bz+EPS or bz+bH<=z+EPS):
                return False
        return True

    while box_pool and fail_count<max_fail:
        # 权重随机选择箱子
        volumes = [dims[0]*dims[1]*dims[2] for _,dims in box_pool]
        weights = np.array(volumes)**alpha
        weights /= weights.sum()
        idx_box = np.random.choice(len(box_pool), p=weights)
        box_type, (L,W,H) = box_pool[idx_box]

        # 随机选候选点
        # 加权随机抽样（不放回）：w 越大被选中的先后概率越高
        if not free_points:
            points = []
        else:
            pts = np.asarray(list(free_points), dtype=float)   # 形状 (N,4) -> [x,y,z,w]
            weights = np.clip(pts[:, 3], 0.0, None)      # 负权置0
            s = weights.sum()
            if s <= 1e-12:
                probs = np.full(len(weights), 1.0/len(weights))
            else:
                probs = weights / s
            order = np.random.choice(len(pts), size=len(pts), replace=False, p=probs)
            points = [tuple(pts[i, :3]) for i in order]

        placed=False
        for fp in points:
            yaws = [0, 90, 180, 270]
            if rdYaws:
                random.shuffle(yaws)
            for yaw in yaws:
                if yaw in [0,180]:
                    Lr,Wr=L,W
                else:
                    Lr,Wr=W,L
                if can_place(fp[0],fp[1],fp[2],Lr,Wr,H):
                    # 放置
                    x,y,z=fp
                    placed_boxes.append([box_type,x,y,z,yaw])
                    occupied.append((x,y,z,Lr,Wr,H))
                    free_points.add((x+Lr,y,z,w_x))
                    free_points.add((x,y+Wr,z,w_y))
                    free_points.add((x,y,z+H,w_z))
                    del box_pool[idx_box]
                    placed=True
                    fail_count=0
                    break
            if placed:
                break
        if not placed:
            fail_count+=1

    return placed_boxes

def boxCorner2boxCenter(boxPosDescriptors, boxesSizes):
    """
    输入:  [seq, x, y, z, yaw]，其中 (x,y,z) 是箱子左下角，yaw 为绕左下角的旋转(度)
    输出:  [seq, x_center, y_center, z_center, yaw]
    """
    centers = []
    for seq, x, y, z, yaw in boxPosDescriptors:
        L, W, H = boxesSizes[seq]
        if yaw == 0 or yaw == 180:
            x_center = x + L / 2
            y_center = y + W / 2
        else:
            x_center = x + W / 2
            y_center = y + L / 2
        centers.append([seq, x_center, y_center, z, yaw])
    return centers

class RectangleArranger2D:
    """
    在以(0,0)为中心的二维场景内摆放若干长方形（不重叠、不越界），并提供可视化。
    ListSq: (N,4) -> [L, W, restrict, qty]
        restrict=1: 角度 ∈ {0, 90, 180, 270}
        restrict=0: 角度 ∈ [0, 360)
    scene_size: [Scene_L, Scene_W]
    """

    def __init__(self, scene_size, seed=None):
        self.Ls = float(scene_size[0])
        self.Ws = float(scene_size[1])
        self.Sx = self.Ls / 2.0
        self.Sy = self.Ws / 2.0
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    # --------- Public API ---------
    def arrange_rects_2d_with_qty(self, ListSq, max_trials_per_rect=2000):
        """
        返回 placements: [[seq, cx, cy, yaw_deg], ...]
        seq 为原始 ListSq 的行号（0-based）
        """
        ListSq = np.asarray(ListSq, dtype=float)

        # 展开数量：[(seq, L, W, restrict), ...]
        expanded = []
        for seq, (L, W, restrict, qty) in enumerate(ListSq):
            qty = int(qty)
            if qty <= 0:
                continue
            expanded.extend([(seq, float(L), float(W), int(restrict)) for _ in range(qty)])

        # 按面积从大到小放置
        order = sorted(expanded, key=lambda t: -(t[1] * t[2]))

        placements_local = []  # (seq, cx, cy, theta, L, W)
        for seq, L, W, restrict in order:
            placed = False
            for _ in range(max_trials_per_rect):
                # 采样角度
                if restrict == 1:
                    yaw_deg = np.random.choice([0, 90, 180, 270])
                else:
                    yaw_deg = np.random.uniform(0.0, 360.0)
                theta = np.deg2rad(yaw_deg)

                # 该角度下 AABB 半宽，保证采样不越界
                c, s = np.cos(theta), np.sin(theta)
                hx = 0.5 * (abs(L * c) + abs(W * s))
                hy = 0.5 * (abs(L * s) + abs(W * c))
                if hx > self.Sx + 1e-9 or hy > self.Sy + 1e-9:
                    continue  # 这个角度下无解，换角

                # 在允许范围内随机采样形心
                cx = np.random.uniform(-self.Sx + hx, self.Sx - hx)
                cy = np.random.uniform(-self.Sy + hy, self.Sy - hy)

                # 与既有矩形做 OBB-OBB 分离轴检测
                ok = True
                for (pseq, px, py, pth, pL, pW) in placements_local:
                    if self._obb_overlap(cx, cy, L, W, theta, px, py, pL, pW, pth):
                        ok = False
                        break
                if ok:
                    placements_local.append((seq, cx, cy, theta, L, W))
                    placed = True
                    break

            if not placed:
                print(f"[WARN] 跳过矩形(seq={seq}, L={L}, W={W}, restrict={restrict})："
                    f"请增大场景或提高 max_trials_per_rect。")
                continue

        # 输出 [seq, cx, cy, yaw_deg]
        placements = [
            [int(seq), float(cx), float(cy), float(np.rad2deg(theta))]
            for (seq, cx, cy, theta, L, W) in placements_local
        ]
        return placements

    # --------- Geometry utils (private) ---------
    @staticmethod
    def _obb_corners(cx, cy, L, W, theta):
        dx, dy = L / 2, W / 2
        local = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]], dtype=float)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
        return (local @ R.T) + np.array([cx, cy])

    @staticmethod
    def _project_polygon(axis, pts):
        proj = pts @ axis
        return proj.min(), proj.max()

    @staticmethod
    def _overlap_1d(a_min, a_max, b_min, b_max):
        return not (a_max < b_min or b_max < a_min)

    def _obb_overlap(self, cx1, cy1, L1, W1, th1, cx2, cy2, L2, W2, th2, eps=1e-9):
        P = self._obb_corners(cx1, cy1, L1, W1, th1)
        Q = self._obb_corners(cx2, cy2, L2, W2, th2)
        ax1 = np.array([np.cos(th1), np.sin(th1)])
        ay1 = np.array([-np.sin(th1), np.cos(th1)])
        ax2 = np.array([np.cos(th2), np.sin(th2)])
        ay2 = np.array([-np.sin(th2), np.cos(th2)])
        axes = [ax1, ay1, ax2, ay2]
        for a in axes:
            a = a / (np.linalg.norm(a) + eps)
            pmin, pmax = self._project_polygon(a, P)
            qmin, qmax = self._project_polygon(a, Q)
            if not self._overlap_1d(pmin, pmax, qmin, qmax):
                return False
        return True

"""2. Preparation"""
"""2.1 Directories Check for saving images and labels"""
if not os.path.exists(path_dir_script):
    os.makedirs(path_dir_script)
    print(f"Created folder: {path_dir_script}")

if not os.path.exists(path_dir_rgb):
    os.makedirs(path_dir_rgb)
    print(f"Created folder: {path_dir_rgb}")
else:
    print(f"Image folder already exists.")

if not os.path.exists(path_dir_depth):
    os.makedirs(path_dir_depth)
    print(f"Created folder: {path_dir_depth}")
else:
    print(f"Depth folder already exists.")

if not os.path.exists(path_dir_label):
    os.makedirs(path_dir_label)
    print(f"Created folder: {path_dir_label}")
else:
    print(f"Label folder already exists.")

"""2.2 Stage Preparation"""
stage = usd.get_context().get_stage()
stage.SetEditTarget(stage.GetRootLayer())
layer = stage.GetRootLayer()

"""2.2.1 Delete redundant objects"""
deleteCopy(stage)

"""2.2.2 Check Scene"""
prim_rdLarge = stage.GetPrimAtPath(uPath_rdLarge)
assert prim_rdLarge.IsValid(), \
    f"RackLarge prim not found at {uPath_rdLarge}"
primList_rdLarge_smBox_original = [stage.GetPrimAtPath(path) for path in uPath_rdLarge_smBoxList_original]
assert all(prim.IsValid() for prim in primList_rdLarge_smBox_original),\
    f"Some small box prims not found in {uPath_rdLarge_smBoxList_original}"
prim_rdLarge_container = stage.GetPrimAtPath(uPath_rdLarge_container)
assert prim_rdLarge_container.IsValid(), \
    f"RackLarge container prim not found at {uPath_rdLarge_container}"
prim_pwbh = stage.GetPrimAtPath(uPath_pwbh)
assert prim_pwbh.IsValid(), \
    f"PWBH prim not found at {uPath_pwbh}"
primList_pwbh_smBox_original = [stage.GetPrimAtPath(path) for path in uPath_pwbh_smBoxList_original]
assert all(prim.IsValid() for prim in primList_pwbh_smBox_original),\
    f"Some small box prims not found in {uPath_pwbh_smBoxList_original}"
prim_forklift = stage.GetPrimAtPath(uPath_forklift)
assert prim_forklift.IsValid(), \
    f"Forklift prim not found at {uPath_forklift}"
prim_dumper = stage.GetPrimAtPath(uPath_dumper)
assert prim_dumper.IsValid(), \
    f"Dumper prim not found at {uPath_dumper}"

"""2.2.3 Set Camera"""
camera = Camera(prim_path=uPath_camera, resolution=(aParam_imgWidth, aParam_imgHeight))

"""3. The Main Loop"""
async def my_task():
    """3.1 Timeline Start"""
    timeline = usd.get_context().get_timeline()
    timeline.play()
    await asyncio.sleep(0.1)
    camera.initialize()
    await asyncio.sleep(0.1)

    """3.2 Annotations Initialization"""
    depth_annotator = AnnotatorRegistry.get_annotator("distance_to_image_plane") # Use "DepthLinearized" for depth annotation
    depth_annotator.attach(camera.get_render_product_path())
    instanceSemantic_annotator = AnnotatorRegistry.get_annotator("instance_segmentation")
    instanceSemantic_annotator.attach(camera.get_render_product_path())
    bounding_box_3d_anno = AnnotatorRegistry.get_annotator("bounding_box_3d")
    bounding_box_3d_anno.attach(camera.get_render_product_path())

    """3.3 Main Loop"""
    while True:
        """3.3.1 Scene Planning"""
        """
        LWRQ:
            - LW: Lang, Width
            - R: Need Rotation Constraint
            - Q: Quantity
        """
        LWRQ_rdLarge = [2.0, 4.0, 1, random.randint(1, aParam_max_rdLarge)]
        LWRQ_pwbh = [2.0, 2.0, 0, random.randint(1, aParam_max_pwbh)]
        LWRQ_forkLift = [2.0, 4.0, 0, random.randint(0, aParam_max_forklift)]
        LWRQ_dumper = [4.0, 1.5, 0, random.randint(0, aParam_max_dumper)]

        createSceneTool = RectangleArranger2D(uParam_sceneSize)
        descriptors_scene = np.array(createSceneTool.arrange_rects_2d_with_qty([LWRQ_rdLarge, LWRQ_pwbh, LWRQ_forkLift, LWRQ_dumper], max_trials_per_rect=2000))
        descriptors_rdLarge = descriptors_scene[descriptors_scene[:, 0] == 0]
        descriptors_pwbh = descriptors_scene[descriptors_scene[:, 0] == 1]
        descriptors_forkLift = descriptors_scene[descriptors_scene[:, 0] == 2]
        descriptors_dumper = descriptors_scene[descriptors_scene[:, 0] == 3]
        print(f"Scene includes {len(descriptors_scene)} objects: {len(descriptors_rdLarge)} rdLarge(s), {len(descriptors_pwbh)} pwbh(s), {len(descriptors_forkLift)} forkLift(s), {len(descriptors_dumper)} dumper(s)")

        """3.3.2 Move the camera"""
        await asyncio.sleep(0.1)
        target_point = np.array([8.0, np.random.uniform(-10, 10), np.random.uniform(1.5, 5)])
        camOri = camPosOri(target_point, aParam_aimedPoint)
        camera.set_world_pose(
            position=target_point,
            orientation=camOri,  
        ) 
        await asyncio.sleep(0.1)

        """3.3.3 Scene Creating"""
        print(f"Timeline is running: {timeline.is_playing()}")
        print(f"time: {timeline.get_current_time()}")

        """3.3.3.1 rdLarge Creation"""
        print(f"Planning {len(descriptors_rdLarge)} rdLarge(s) with descriptors: {descriptors_rdLarge}")
        sdf_rdLarge = Sdf.Path(uPath_rdLarge)
        if not prim_rdLarge.IsActive():
            print(f"{uPath_rdLarge} deactivated, activating...")
            prim_rdLarge.SetActive(True)
        primList_rdLarge_created = []
        primListList_rdLarge_smBox_created = []
        primListList_rdLarge_container_created = []
        for i, desc in enumerate(descriptors_rdLarge):
            px, py, pyaw = desc[1], desc[2], desc[3]
            px = uParam_transScale * px
            py = uParam_transScale * py
            quantity_rdLarge_smBox = [random.randint(0, aParam_max_rdLarge_smBox[0]),
                                    random.randint(0, aParam_max_rdLarge_smBox[1]), 
                                    random.randint(0, aParam_max_rdLarge_smBox[2])]
            """3.3.3.1.1 rdLarge smBox Heaps Creation"""
            descriptors_rdLarge_smBox = pack_boxes_weighted_random(
                uParam_rdLarge_Space, uParam_smBoxSizeList, quantity_rdLarge_smBox, 
                support_threshold=uParam_supportThreshold, 
                alpha=aParam_distriBoxWeightAlpha,
                max_fail=50,
                w_x=1, w_y=1, w_z=0.5, rdYaws=True
            )
            descriptors_rdLarge_smBox_center = boxCorner2boxCenter(descriptors_rdLarge_smBox, uParam_smBoxSizeList)
            primList_rdLarge_smBox_created = []
            for prim in primList_rdLarge_smBox_original:
                imageable = UsdGeom.Imageable(prim)
                imageable.MakeInvisible()
            for j, bPDc in enumerate(descriptors_rdLarge_smBox_center):
                seq, x, y, z, yaw = bPDc
                x_w, y_w, z_w, _ = uParam_rdLarge_mtxTrans @ np.array([x, y, z, 1.0])
                prim_original = primList_pwbh_smBox_original[seq]
                path_new = f"/World/RackLarge_A1/Box_{seq}_{j}"
                prim_new = stage.OverridePrim(path_new)
                prim_new.GetReferences().AddReference(assetPath="", primPath=prim_original.GetPath())
                if prim_new.IsValid():
                    imageable = UsdGeom.Imageable(prim_new)
                    imageable.MakeVisible()
                    xform = UsdGeom.Xformable(prim_new)
                    xform.ClearXformOpOrder()
                    xform_trans = xform.AddTranslateOp(opSuffix="")
                    xform_rot = xform.AddRotateXYZOp(opSuffix="")
                    xform_trans.Set(value=Gf.Vec3d(x_w, y_w, z_w))
                    xform_rot.Set(value=Gf.Vec3d(0, 0, yaw + uParam_rdLarge_yaw0))
                    primList_rdLarge_smBox_created.append(prim_new)
                    print(f"Placed box_{seq}_{j} at ({x_w}, {y_w}, {z_w}) with yaw {yaw} degrees.")
            primListList_rdLarge_smBox_created.append(primList_rdLarge_smBox_created)

            """3.3.3.1.2 rdLarge container Creation"""
            imageable = UsdGeom.Imageable(prim_rdLarge_container)
            imageable.MakeInvisible()
            LWRQ_rdLarge_container = [40, 30, 0, random.randint(1, aParam_max_rdLarge_container)]
            createContainerTool = RectangleArranger2D(uParam_transScale*np.array(uParam_rdLarge_Space[:2]))
            descriptors_rdLarge_container = createContainerTool.arrange_rects_2d_with_qty(
                LWRQ_rdLarge_container, max_trials_per_rect=2000)
            primList_rdLarge_container_created = []
            for j, bPDc in enumerate(descriptors_rdLarge_container):
                _, x_w, y_w, yaw = bPDc
                z_w = uParam_transScale * uParam_rdLarge_Height_2
                path_new = f"/World/RackLarge_A1/Container_{j}"
                prim_new = stage.OverridePrim(path_new)
                primList_rdLarge_container_created.append(prim_new)
                prim_new.GetReferences().AddReference(assetPath="", primPath=uPath_rdLarge_container)
                imageable = UsdGeom.Imageable(prim_new)
                imageable.MakeVisible()
                xform = UsdGeom.Xformable(prim_new)
                xform.ClearXformOpOrder()
                xform_trans = xform.AddTranslateOp(opSuffix="")
                xform_rot = xform.AddRotateXYZOp(opSuffix="")
                xform_trans.Set(value=Gf.Vec3d(x_w, y_w, z_w))
                xform_rot.Set(value=Gf.Vec3d(0, 0, yaw))
                print(f"Placed container_{j} at ({x_w}, {y_w}, {z_w}) with yaw {yaw} degrees.")
            primListList_rdLarge_container_created.append(primList_rdLarge_container_created)

        """3.3.3.1.3 rdLarge Main Creation"""
        
