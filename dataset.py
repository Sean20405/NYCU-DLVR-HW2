import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        self.images = coco_data['images']
        self.annotations = coco_data['annotations']
        self.categories = coco_data['categories']

        # Create image_id -> annotations mapping
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Create image_id -> image_filename mapping
        self.img_id_to_filename = {
            img['id']: img['file_name'] for img in self.images
        }

        print(
            f'Loaded {len(self.images)} images and '
            f'{len(self.annotations)} annotations'
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.root_dir, self.img_id_to_filename[img_id])

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])

        boxes = []
        labels = []
        for ann in anns:
            # COCO format is [x, y, width, height]
            # Convert to [x_min, y_min, x_max, y_max]
            x, y, width, height = ann['bbox']
            boxes.append([x, y, x + width, y + height])
            labels.append(ann['category_id'])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dict
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img_id])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


class TestDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.images = [
            f for f in os.listdir(root_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]

        print(f'Loaded {len(self.images)} test images')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        # Extract image_id from filename
        image_id = int(os.path.splitext(self.images[idx])[0])

        return img, image_id
