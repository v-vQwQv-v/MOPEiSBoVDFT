from pxr import Usd, UsdGeom
import omni.usd

stage = omni.usd.get_context().get_stage()

prim = stage.GetPrimAtPath("/World/warehouse_with_forklifts/SM_CardBoxC_01")

# 使用 UsdGeom.Imageable 设置可见性
imageable = UsdGeom.Imageable(prim)
imageable.MakeInvisible()

imageable.MakeVisible()


