# 此脚本验证箱子位姿变化以及标签导出
import numpy as np
import omni.timeline
import asyncio
import omni.usd as usd
import pxr.Gf as Gf
from pxr import UsdGeom, Gf, Usd
import numpy as np
import omni.timeline

timeline = omni.timeline.get_timeline_interface()
timeline.play()

# 第二步
stage = usd.get_context().get_stage()
box_prim = stage.GetPrimAtPath("/World/warehouse_with_forklifts/SM_CardBoxC_Copy_4")

# 第三步
imageable = UsdGeom.Imageable(box_prim)
bound = imageable.ComputeWorldBound(timeline.get_current_time(), UsdGeom.Tokens.default_)
bound_range = bound.ComputeAlignedBox()
print(f"bound_range: {bound_range}")

# 第四步
min_corner = bound_range.GetMin()  # 世界坐标系下最小角点[x, y, z]
max_corner = bound_range.GetMax()  # 世界坐标系下最大角点[x, y, z]
print(f"min_corner: {min_corner}, max_corner: {max_corner}")

bbox_prim = stage.DefinePrim("/World/AABB_Visual", "Cube")
bbox_prim.GetAttribute("size").Set(1.0)
bbox_prim.GetAttribute("primvars:displayColor").Set([(1, 0, 0)])  

# 第五步：画框
center = (min_corner + max_corner) / 2
extents = max_corner - min_corner
xform = UsdGeom.Xformable(bbox_prim)
xform.AddTranslateOp().Set(center)
xform.AddScaleOp().Set(extents)

# 第六步：保存位姿


