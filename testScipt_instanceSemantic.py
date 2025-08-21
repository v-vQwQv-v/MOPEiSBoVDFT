# 深度图需要分部执行
import asyncio
from isaacsim.sensors.camera import Camera
from omni.replicator.core import AnnotatorRegistry
import omni.timeline
import cv2
import numpy as np
import json

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
instanceSemantic_annotator = AnnotatorRegistry.get_annotator("instance_segmentation")
instanceSemantic_annotator.attach(camera.get_render_product_path())

# 第五段
instanceSemantic_data = instanceSemantic_annotator.get_data() 
print(f"semantic_data: {instanceSemantic_data}")

# 第六段
# id_to_labels = instanceSemantic_data['info']['idToLabels']
# mask = []
# for sem_id, label_info in id_to_labels.items():
#     for i in range(20):
#         if label_info == f'/World/warehouse_with_forklifts/SM_CardBoxC_Copy_{i}/SM_CardBoxC_01':
#             print(f"box_id: {sem_id}")
#             mask.append(instanceSemantic_data['data'] == int(sem_id))

# instanceSemantic_image = cv2.cvtColor(camera.get_rgb(), cv2.COLOR_BGR2RGB)

# for j in range(len(mask)):
#     instanceSemantic_image[mask[j] == 1] = np.random.randint(0, 256, size=3)

# cv2.imwrite(f"{script_dir}/test_instanceSemantic.png", instanceSemantic_image)

# 第六段
semantic_id_map = instanceSemantic_data['data']  # shape: (H, W), dtype=uint32
height, width = semantic_id_map.shape
id_to_semantics = instanceSemantic_data['info']['idToSemantics']
target_classes = {"blockpallet", "container", "cardbox_a", "cardbox_c", "cardbox_d", "rack", "dumper", "forklift"}
id_to_semantics = instanceSemantic_data['info']['idToSemantics']
valid_ids = [
    int(sem_id) for sem_id, sem in id_to_semantics.items()
    if sem.get('class') in target_classes
]
id_to_color = {}
rng = np.random.default_rng(seed=42)
for sem_id_str, sem_data in id_to_semantics.items():
    sem_id = int(sem_id_str)
    class_name = sem_data.get("class", "").lower()
    if class_name in target_classes:
        id_to_color[sem_id] = rng.integers(0, 256, size=3, dtype=np.uint8)
mask_color_image = np.zeros((height, width, 3), dtype=np.uint8)
for sem_id, color in id_to_color.items():
    mask = (semantic_id_map == sem_id)
    mask_color_image[mask] = color
rgb = camera.get_rgb()
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(rgb, 0.5, mask_color_image, 0.5, 0)
cv2.imwrite(f"{script_dir}/semantic_overlay_target_classes.png", overlay)
cv2.imwrite(f"{script_dir}/semantic_mask_target_classes.png", mask_color_image)
print("Valid IDs for target classes:", valid_ids)
