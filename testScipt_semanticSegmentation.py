# 深度图需要分部执行
import asyncio
from isaacsim.sensors.camera import Camera
from omni.replicator.core import AnnotatorRegistry
import omni.timeline
import cv2
import numpy as np

script_dir = "E:/VScode/VSworkspace/pyworkspace/isaacsimpy"
camera = Camera(prim_path="/World/Camera_0", resolution=(1024, 512))

# 第二段
camera.initialize()

timeline = omni.timeline.get_timeline_interface()
timeline.play()

# 第三段
camera.add_motion_vectors_to_frame()
print(f"render_product_path: {camera.get_render_product_path()}")

# 第四段
semantic_annotator = AnnotatorRegistry.get_annotator("semantic_segmentation")
semantic_annotator.attach(camera.get_render_product_path())

# 第五段
semantic_data = semantic_annotator.get_data() 

print(f"semantic_data: {semantic_data}")


# 第六段
id_to_labels = semantic_data['info']['idToLabels']
box_id = None
for sem_id, label_info in id_to_labels.items():
    if label_info['class'] == 'box':
        box_id = int(sem_id)  # 转换为整数类型
        break
mask = (semantic_data['data'] == box_id).astype(np.uint8)

print(f"mask: {mask.shape}")

semantic_image = cv2.cvtColor(camera.get_rgb(), cv2.COLOR_BGR2RGB)
semantic_image[mask == 1] = [255, 0, 0] 

cv2.imwrite(f"{script_dir}/test_semantic.png", semantic_image)