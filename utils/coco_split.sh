#!/usr/bin/env bash

# 1) Dump the filenames from your split JSON
jq -r '.images[].file_name' /scratch/rob498w25_class_root/rob498w25_class/radithya/ViTPose/data/coco_subtrain/annotations/person_keypoints_subtrain2017.json \
  > subtrain_files.txt

# 2) Copy only those files into your subtrain folder
rsync -av --files-from=subtrain_files.txt \
      /scratch/rob498w25_class_root/rob498w25_class/radithya/ViTPose/data/coco/train2017/ \
      /scratch/rob498w25_class_root/rob498w25_class/radithya/ViTPose/data/coco_subtrain/train2017/
