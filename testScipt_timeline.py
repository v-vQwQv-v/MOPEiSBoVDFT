# 时间线控制开始结束
from omni.isaac.core import World
import omni.timeline

world = World()
world.step()  # 必须至少执行一次step()
timeline = omni.timeline.get_timeline_interface()
current_time = timeline.get_current_time()
print(f"Omniverse时间轴时间: {current_time:.3f}秒")

# import time
# time.sleep(5)