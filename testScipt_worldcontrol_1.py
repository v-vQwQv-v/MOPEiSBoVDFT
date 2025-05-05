# 在实时脚本中需要异步编程
import omni.timeline
import asyncio
from omni.isaac.sensor import Camera

camera = Camera(prim_path="/World/Camera_0", resolution=(1024, 512))

timeline = omni.timeline.get_timeline_interface()
timeline.play() 
 

async def my_task():
    while True:
        # Wait for the next frame to be ready
        await asyncio.sleep(1)
        print(f"Timeline is running: {timeline.is_playing()}")
        print(f"time: {timeline.get_current_time()}")
        await asyncio.sleep(1)
        if timeline.get_current_time() >= 10:
            print("Stopping timeline after 10 seconds.")
            break

asyncio.ensure_future(my_task())
    
    




