import omni.timeline
import asyncio
import omni.usd as usd
import random
import pxr.Gf as Gf
import numpy as np
from pxr import Usd, UsdGeom, Sdf, UsdUtils
import omni.kit.commands as cmds

"""param"""
rdLargeSpace = [1.84, 3.7, 1.84] # 货架空间大小 [L, W, H]，单位为米
transScale = 100 # 变换缩放比例，箱子的位置坐标是以米为单位的，而在USD中通常使用厘米，所以这里设置为100
rdLargeHeight_1 = 0.5 # 架子第一层高度，单位为米
rdLargeHeight_2 = 2.38 # 架子第二层高度，单位为米
tolerance = 0.05 # 容忍度，单位为米
support_threshold = 0.8 # 支撑阈值，比例
mtxTrans_vitual2Sim_rdLarge = transScale* np.array([[1,     0,  0, -rdLargeSpace[0]/2 + tolerance],
                                            [0,    1,  0, -rdLargeSpace[1]/2 + tolerance],
                                            [0,     0,  1, rdLargeHeight_1],
                                            [0,     0,  0, 1]], dtype=np.float32)

yaw_0 = 90

boxes1Sizes = [[0.26, 0.35, 0.16], [0.52, 0.52, 0.26], [0.52, 0.73, 0.52]] # 三种箱子的尺寸
box1Quantities = [30, 20, 10] # 每种箱子的数量
box2Quantities = [20, 10, 5]  # 每种箱子的数量
alpha = 2.0  # 权重分配：大箱子权重高


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

"""Start"""
stage = usd.get_context().get_stage()
stage.SetEditTarget(stage.GetRootLayer())
flat = UsdUtils.FlattenLayerStack(stage)
layer = stage.GetRootLayer()

"""BoxHeapsCreation"""
originalRdLargePath = Sdf.Path("/World/RackLarge_A1")
originalRdLargePrim = stage.GetPrimAtPath(str(originalRdLargePath))
if not originalRdLargePrim.IsActive():
        print(f"{originalRdLargePath} deactivated, activating...")
        originalRdLargePrim.SetActive(True)

listOriginalBox1Path = [
    "/World/RackLarge_A1/Cardbox_D2/Cardbox_D2",
    "/World/RackLarge_A1/Cardbox_C3/Cardbox_C3",
    "/World/RackLarge_A1/Cardbox_A3/Cardbox_A3"
]
box1PosDescriptors = pack_boxes_weighted_random(
    rdLargeSpace, boxes1Sizes, box1Quantities, 
    support_threshold=support_threshold, 
    alpha=alpha,
    max_fail=50,
    w_x=1, w_y=1, w_z=0.5, rdYaws=True
)
box1PosDescriptors_center = boxCorner2boxCenter(box1PosDescriptors, boxes1Sizes)
listOriginalBox1Prim = []
for path in listOriginalBox1Path:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim at path '{path}' is invalid or does not exist.")
    listOriginalBox1Prim.append(prim)
    imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
    imageable.MakeInvisible()
listCreatedBox1Prim = []
for i, bPDc in enumerate(box1PosDescriptors_center):
    seq, x, y, z, yaw = bPDc
    x_w, y_w, z_w, _ = mtxTrans_vitual2Sim_rdLarge @ np.append([x, y, z], 1)
    OriginalBoxPrim = listOriginalBox1Prim[seq]
    newBox1Path = f"/World/RackLarge_A1/Box_{seq}_{i}"
    newBox1Prim = stage.OverridePrim(newBox1Path)
    newBox1Prim.GetReferences().AddReference(assetPath="", primPath=OriginalBoxPrim.GetPath())
    if newBox1Prim.IsValid():
        box1Imgable = UsdGeom.Imageable(newBox1Prim)
        box1Imgable.MakeVisible()
        box1Xform = UsdGeom.Xformable(newBox1Prim)
        box1Xform.ClearXformOpOrder()
        box1Xform_translate_op = box1Xform.AddTranslateOp(opSuffix="")
        box1Xform_rotate_op = box1Xform.AddRotateXYZOp(opSuffix="")
        box1Xform_translate_op.Set(Gf.Vec3d(x_w, y_w, z_w))
        box1Xform_rotate_op.Set(Gf.Vec3f(0, 0, yaw_0 + yaw))
        print(f"Placed box_{seq}_{i} at ({x_w}, {y_w}, {z_w}) with yaw {yaw} degrees.")
        listCreatedBox1Prim.append(newBox1Prim)

"""BlueBoxesCreation"""
originalBlueBoxPath = "/World/RackLarge_A1/Container_B09_40x30x22cm_PR_V_NVD_01"
originalBlueBoxPrim = stage.GetPrimAtPath(originalBlueBoxPath)
if not originalBlueBoxPrim.IsValid():
    raise RuntimeError(f"Prim at path '/World/RackLarge_A1/Container_B09_40x30x22cm_PR_V_NVD_01' is invalid or does not exist.")
else:
    imageable = UsdGeom.Imageable(originalBlueBoxPrim)
    imageable.MakeInvisible()
    # imageable.MakeVisible()
blueBoxParam = [[40, 30, 0, random.randint(1, 10)]]  # [L, W, restrict, qty]
createBlueBoxTool = RectangleArranger2D(transScale*np.array(rdLargeSpace[:2]))
blueBoxPosDescriptors = createBlueBoxTool.arrange_rects_2d_with_qty(blueBoxParam, max_trials_per_rect=2000) # [seq, cx, cy, yaw_deg]*N
listCreatedBlueBoxPrim = []
if len(blueBoxPosDescriptors) != 0:
    for i, bBPDc in enumerate(blueBoxPosDescriptors):
        _, x_w, y_w, yaw = bBPDc
        z_w = transScale * rdLargeHeight_2
        newBlueBoxPath = f"/World/RackLarge_A1/BlueBox_{i}"
        newBlueBoxPrim = stage.OverridePrim(newBlueBoxPath)
        newBlueBoxPrim.GetReferences().AddReference(assetPath="", primPath=originalBlueBoxPath)
        if newBlueBoxPrim.IsValid():
            blueboxImgable = UsdGeom.Imageable(newBlueBoxPrim)
            blueboxImgable.MakeVisible()
            blueboxXform = UsdGeom.Xformable(newBlueBoxPrim)
            blueboxXform.ClearXformOpOrder()
            blueboxXform_translate_op = blueboxXform.AddTranslateOp(opSuffix="")
            blueboxXform_rotate_op = blueboxXform.AddRotateXYZOp(opSuffix="")
            blueboxXform_translate_op.Set(Gf.Vec3d(x_w, y_w, z_w))
            blueboxXform_rotate_op.Set(Gf.Vec3f(0, 0, yaw))
            print(f"Placed blueBox_{i} at ({x_w}, {y_w}, {z_w}) with yaw {yaw} degrees.")
            listCreatedBlueBoxPrim.append(newBlueBoxPrim)

"""Copy RdLarge"""
newRdLargePath = Sdf.Path("/World/RackLarge_A1_Copy")
if stage.GetPrimAtPath(str(newRdLargePath)).IsValid():
    stage.RemovePrim(newRdLargePath)
cmds.execute("CopyPrim", path_from=originalRdLargePath, path_to=newRdLargePath)
newRdLargePrim = stage.GetPrimAtPath(str(newRdLargePath))
rdLargeXform = UsdGeom.Xformable(newRdLargePrim)
rdLargeXform.ClearXformOpOrder()
rdLargeXform_scale_op = rdLargeXform.AddScaleOp(opSuffix="")
rdLargeXform_translate_op = rdLargeXform.AddTranslateOp(opSuffix="")
rdLargeXform_rotate_op = rdLargeXform.AddRotateXYZOp(opSuffix="")
rdLargeXform_scale_op.Set(Gf.Vec3d(0.01, 0.01, 0.01))
rdLargeXform_translate_op.Set(Gf.Vec3d(100, 200, 0))
rdLargeXform_rotate_op.Set(Gf.Vec3f(0, 0, 30))

"""repeat k times"""

"""PwbhesCreation"""
listOriginalBox2Path = [
    "/World/WarehousePile_A6/Cardbox_D2",
    "/World/WarehousePile_A6/Cardbox_C1",
    "/World/WarehousePile_A6/Cardbox_A3"
]
pwbhSpace = [1.70, 1.90, 1.80] # 托盘空间大小 [L, W, H]
pwbhHeight = 0.21
mtxTrans_vitual2Sim_pwbh = transScale * np.array([[1, 0, 0, -pwbhSpace[0]/2 + tolerance],
                                                  [0, 1, 0, -pwbhSpace[1]/2 + tolerance],
                                                  [0, 0, 1, pwbhHeight],
                                                  [0, 0, 0, 1]])
box2PosDescriptors = pack_boxes_weighted_random(
    pwbhSpace, boxes1Sizes, box2Quantities, 
    support_threshold=support_threshold, 
    alpha=alpha,
    max_fail=50,
    w_x=1, w_y=1, w_z=0.5, rdYaws=True
)
box2PosDescriptors_center = boxCorner2boxCenter(box2PosDescriptors, boxes1Sizes)
listOriginalBox2Prim = []
for path in listOriginalBox2Path:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim at path '{path}' is invalid or does not exist.")
    listOriginalBox2Prim.append(prim)
    imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
    imageable.MakeInvisible()
listCreatedBox2Prim = []
for i, bPDc in enumerate(box2PosDescriptors_center):
    seq, x, y, z, yaw = bPDc
    x_w, y_w, z_w, _ = mtxTrans_vitual2Sim_pwbh @ np.append([x, y, z], 1)
    OriginalBoxPrim = listOriginalBox2Prim[seq]
    newBox2Path = f"/World/WarehousePile_A6/Box_{seq}_{i}"
    newBox2Prim = stage.OverridePrim(newBox2Path)
    newBox2Prim.GetReferences().AddReference(assetPath="", primPath=OriginalBoxPrim.GetPath())
    if newBox2Prim.IsValid():
        box2Imgable = UsdGeom.Imageable(newBox2Prim)
        box2Imgable.MakeVisible()
        box2Xform = UsdGeom.Xformable(newBox2Prim)
        box2Xform.ClearXformOpOrder()
        box2Xform_translate_op = box2Xform.AddTranslateOp(opSuffix="")
        box2Xform_rotate_op = box2Xform.AddRotateXYZOp(opSuffix="")
        box2Xform_translate_op.Set(Gf.Vec3d(x_w, y_w, z_w))
        box2Xform_rotate_op.Set(Gf.Vec3f(0, 0, yaw_0 + yaw))
        print(f"Placed box_{seq}_{i} at ({x_w}, {y_w}, {z_w}) with yaw {yaw} degrees.")
        listCreatedBox2Prim.append(newBox2Prim)

"""repeat k times"""

"""DumpersCreation: use the same copy-method like rdLarge"""

"""ForksLiftCreation: use the same copy-method like rdLarge"""

"""Delete created boxes"""
for box in listCreatedBox1Prim:
    if box.IsValid():
        box.GetStage().RemovePrim(box.GetPath())
        print(f"Deleted created box at {box.GetPath()}")
for box in listCreatedBlueBoxPrim:
    if box.IsValid():
        box.GetStage().RemovePrim(box.GetPath())
        print(f"Deleted created blue box at {box.GetPath()}")
for box in listCreatedBox2Prim:
    if box.IsValid():
        box.GetStage().RemovePrim(box.GetPath())
        print(f"Deleted created box at {box.GetPath()}")
for path in listOriginalBox1Path:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
        imageable.MakeVisible()
        print(f"Made original box at {path} visible.")
for path in listOriginalBox2Path:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
        imageable.MakeVisible()
        print(f"Made original box at {path} visible.")
if originalBlueBoxPrim.IsValid():
    imageable = UsdGeom.Imageable(originalBlueBoxPrim)
    imageable.MakeVisible()
    print(f"Made original blue box at {originalBlueBoxPrim.GetPath()} visible.")




