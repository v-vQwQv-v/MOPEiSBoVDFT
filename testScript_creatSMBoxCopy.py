import omni
import random
import omni.usd as usd
from pxr import UsdGeom, UsdUtils
import pxr.Sdf as Sdf
import pxr.Gf as Gf
import pxr.Usd as Usd

import os
import sys

script_dir = "E:/VScode/VSworkspace/pyworkspace/isaacsimpy"
import sys
sys.path.insert(0, str(script_dir))
import parameter as param



stage = usd.get_context().get_stage()
original_box_prim = stage.GetPrimAtPath("/World/warehouse_with_forklifts/SM_CardBoxC_01")
if not original_box_prim.IsValid():
    print("Error: Original box not found!")

for i in range(param.num_random_boxes):
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

