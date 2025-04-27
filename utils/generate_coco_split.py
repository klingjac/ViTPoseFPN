import json
import random
from pathlib import Path

# Paths
orig_ann = Path('data/coco/annotations/person_keypoints_train2017.json')
sub_ann  = Path('data/coco_subtrain/annotations/person_keypoints_subtrain2017.json')
sub_img_dir = Path('data/coco_subtrain/train2017')

# Create dirs
sub_ann.parent.mkdir(parents=True, exist_ok=True)
sub_img_dir.mkdir(parents=True, exist_ok=True)

# Load original annotations
coco = json.loads(orig_ann.read_text())
image_list = coco['images']
ann_list   = coco['annotations']

# Sample 10% of images
random.seed(42)
sample_size = max(1, int(len(image_list) * 0.10))
sub_images = random.sample(image_list, sample_size)
sub_ids    = {img['id'] for img in sub_images}

# Filter annotations
sub_anns = [ann for ann in ann_list if ann['image_id'] in sub_ids]

# Build new COCO dict
coco_sub = {
    'info'      : coco['info'],
    'licenses'  : coco['licenses'],
    'categories': coco['categories'],
    'images'    : sub_images,
    'annotations': sub_anns
}

# Write it out
sub_ann.write_text(json.dumps(coco_sub, indent=2))
print(f"→ Wrote {len(sub_images)} images and {len(sub_anns)} annotations to {sub_ann}")
