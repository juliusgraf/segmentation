import os
import json
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

def generate_masks(images_dir, masks_dir, annotations_file):
    # Load COCO annotations
    coco = COCO(annotations_file)

    # Create masks directory if it doesn't exist
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    for img_file in os.listdir(images_dir):
        # Get image ID from the file name
        img_id = int(os.path.splitext(img_file)[0].split('_')[-1])
        
        # Load image to get its size
        img_path = os.path.join(images_dir, img_file)
        image = Image.open(img_path)
        img_width, img_height = image.size

        # Create an empty mask
        mask = np.zeros((img_height, img_width))

        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Generate mask
        for ann in anns:
            mask = np.maximum(mask, coco.annToMask(ann))

        # Save mask as GIF
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_path = os.path.join(masks_dir, os.path.splitext(img_file)[0] + '_mask.gif')
        mask_image.save(mask_path)

# Paths to images and annotations
train_images_dir = 'coco/images/train2017'
val_images_dir = 'coco/images/val2017'
annotations_file_val = 'coco/annotations/instances_val2017.json'
annotations_file_train = 'coco/annotations/instances_train2017.json'

if __name__ == "__main__":
    # Generate and save masks
    generate_masks(train_images_dir, 'coco/images/train_masks', annotations_file_train)
    generate_masks(val_images_dir, 'coco/images/val_masks', annotations_file_val)