import omni
import random
import omni.usd as usd
from pxr import UsdGeom, UsdUtils
import pxr.Sdf as Sdf
import pxr.Gf as Gf
import pxr.Usd as Usd

import os

script_dir = "E:/VScode/VSworkspace/pyworkspace/isaacsimpy"
import sys
sys.path.insert(0, str(script_dir))
import parameter as param


stage = usd.get_context().get_stage()
layer = stage.GetRootLayer()

# path_boxToDelete = [Sdf.Path(f"/World/warehouse_with_forklifts/SM_CardBoxC_Copy_{i}") for i in range(5)]
stage_box_copy = [stage.GetPrimAtPath(f"/World/warehouse_with_forklifts/SM_CardBoxC_Copy_{i}") for i in range(param.num_random_boxes)]

for i, box_prim in enumerate(stage_box_copy):
    if box_prim.IsValid():
        print(f"Box {i}: {box_prim.GetPath()} is need to be deleted")
    else:
        print(f"Box {i}: Path not found")

for box_prim in stage_box_copy:
    if box_prim.IsValid():
        stage.RemovePrim(box_prim.GetPath())
        print(f"Deleted {box_prim.GetPath()}")