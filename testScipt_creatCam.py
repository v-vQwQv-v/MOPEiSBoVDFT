from isaacsim.sensors.camera import Camera
import omni.timeline
import cv2

timeline = omni.timeline.get_timeline_interface()
script_dir = "E:/VScode/VSworkspace/pyworkspace/isaacsimpy"
camera = Camera(prim_path="/World/Camera_0", resolution=(1024, 512))

camera.initialize()

camera.add_motion_vectors_to_frame()
rgb_image = camera.get_rgba()
print(f" {rgb_image}")

bgr_image = cv2.cvtColor(rgb_image[..., :3], cv2.COLOR_RGB2BGR)
cv2.imwrite(f"{script_dir}/output.png", bgr_image)



