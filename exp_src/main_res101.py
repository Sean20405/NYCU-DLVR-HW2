import os
import json
import time
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet101_Weights
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms as T

from utils import (
    parse_args,
    calculate_mAP,
    plot_img_bbox
)

# !!! Remember to change this for each experiment !!!
model_name = 'FasterRCNN_Res101'

args = parse_args(model_name)

# Set GPU ID
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)


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


class FasterRCNN_Res101:
    def __init__(self, args, data_dir='data', num_classes=None,
                 lr=5e-5, model_name=None):
        """
        Initialize the Faster R-CNN model for digit recognition.

        Args:
            data_dir: Directory containing the dataset
            num_classes: Number of classes (including background)
            batch_size: Batch size for training
            learning_rate: Initial learning rate
        """
        if model_name is None:
            self.model_name = 'FastRCNN' + time.strftime("%m%d-%H%M%S")
        else:
            self.model_name = model_name
        self.args = args
        self.data_dir = data_dir
        self.batch_size = args.batch_size
        self.lr = lr

        print(f'Model Name: {model_name}')

        # Data directories
        self.train_dir = os.path.join(data_dir, 'train')
        self.valid_dir = os.path.join(data_dir, 'valid')
        self.test_dir = os.path.join(data_dir, 'test')
        self.train_json = os.path.join(data_dir, 'train.json')
        self.valid_json = os.path.join(data_dir, 'valid.json')

        # Determine number of classes if not provided
        if num_classes is None:
            with open(self.train_json, 'r') as f:
                train_data = json.load(f)
            categories = train_data['categories']
            self.num_classes = len(categories) + 1  # +1 for background class
        else:
            self.num_classes = num_classes

        # Initialize model, optimizer, and scheduler
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = self._get_model()
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=0
        )

        # Best model tracking
        self.ckpt_dir = 'ckpt'
        self.best_accuracy = 0.0
        self.best_map = 0.0

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.best_model_path = os.path.join(
            self.ckpt_dir, f'{self.model_name}_best.pth'
        )
        self.best_map_model_path = os.path.join(
            self.ckpt_dir, f'{self.model_name}_best_map.pth'
        )

        # Initialize datasets and dataloaders
        self._initialize_datasets()

        self.writer = SummaryWriter(f'runs/exp/{self.model_name}')

        # mAP evaluation parameters
        self.iou_thresholds = np.arange(
            0.5, 1.0, 0.05
        )  # [0.5, 0.55, ..., 0.95]
        self.score_threshold = 0.5

    def _get_model(self):
        """Create and return the Faster R-CNN model"""
        # Load pre-trained model
        weights = ResNet101_Weights.IMAGENET1K_V2
        backbone = resnet_fpn_backbone(
            backbone_name='resnet101',
            weights=weights
        )
        model = FasterRCNN(backbone, num_classes=self.num_classes)

        return model

    def _get_transform(self, train):
        """Get transformation pipeline"""
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))

        return T.Compose(transforms)

    def _initialize_datasets(self):
        """Initialize datasets and dataloaders"""
        # Create datasets
        if self.args.train:
            self.train_dataset = CocoDataset(
                self.train_dir, self.train_json,
                transforms=self._get_transform(train=True)
            )
            self.valid_dataset = CocoDataset(
                self.valid_dir, self.valid_json,
                transforms=self._get_transform(train=False)
            )

            # Create data loaders
            self.train_loader = DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=self.args.num_workers,
                collate_fn=lambda x: tuple(zip(*x))
            )
            self.valid_loader = DataLoader(
                self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=lambda x: tuple(zip(*x))
            )
        elif self.args.test or args.show_test:
            self.test_dataset = TestDataset(
                self.test_dir,
                transforms=T.Compose([T.ToTensor()])
            )
            self.test_loader = DataLoader(
                self.test_dataset, batch_size=self.batch_size*2, shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=lambda x: tuple(zip(*x))
            )

    def train_one_epoch(self, epoch):
        """Train the model for one epoch"""
        self.model.train()

        running_loss = 0.0
        running_loss_dict = {}

        for i, (images, targets) in enumerate(tqdm(self.train_loader)):
            images = list(image.to(self.device) for image in images)
            targets = [
                {k: v.to(self.device) for k, v in t.items()}
                for t in targets
            ]

            # Forward pass
            loss_dict = self.model(images, targets)

            # Record individual losses
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in running_loss_dict:
                    running_loss_dict[loss_name] = 0.0
                running_loss_dict[loss_name] += loss_value.item()

            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            running_loss += losses.item()

            # Log training loss and lr
            global_step = epoch * len(self.train_loader) + i
            self.writer.add_scalar(
                'Train/Training Loss',
                losses.item(),
                global_step
            )
            self.writer.add_scalar(
                'Train/Learning Rate',
                self.optimizer.param_groups[0]['lr'],
                global_step
            )
            for loss_name, loss_value in loss_dict.items():
                self.writer.add_scalar(
                    f'Train/{loss_name}',
                    loss_value.item(),
                    global_step
                )

        epoch_loss = running_loss / len(self.train_loader)
        self.writer.add_scalar('Train/Epoch Loss', epoch_loss, epoch)

        return epoch_loss

    def evaluate(self, epoch):
        """Evaluate the model on the validation set"""
        self.model.eval()

        score_sum = 0.0
        all_predictions = []
        all_targets = []

        # Store predictions and ground truths for each image
        image_pred = {}
        image_gt = {}

        all_image_ids = set()

        with torch.no_grad():
            for images, targets in self.valid_loader:
                images = list(image.to(self.device) for image in images)
                targets = [
                    {k: v.to(self.device) for k, v in t.items()}
                    for t in targets
                ]

                # Forward pass for predictions
                outputs = self.model(images)
                for i, (output, target) in enumerate(zip(outputs, targets)):
                    image_id = target['image_id'].item()
                    all_image_ids.add(image_id)

                    # Get predicted boxes, scores, and labels
                    pred_boxes = output['boxes'].cpu()
                    pred_scores = output['scores'].cpu()
                    pred_labels = output['labels'].cpu()

                    # Get ground truth boxes and labels
                    gt_boxes = target['boxes'].cpu()
                    gt_labels = target['labels'].cpu()

                    # Filter predictions by score threshold
                    keep = pred_scores >= self.score_threshold
                    pred_boxes = pred_boxes[keep]
                    pred_scores = pred_scores[keep]
                    pred_labels = pred_labels[keep]

                    # Store predictions and targets for this image
                    score_sum += pred_scores.sum().item()
                    all_predictions.append({
                        'boxes': pred_boxes,
                        'scores': pred_scores,
                        'labels': pred_labels
                    })
                    all_targets.append({
                        'boxes': gt_boxes,
                        'labels': gt_labels
                    })

                    # Store predictions for each image
                    image_pred[image_id] = {
                        'boxes': [],
                        'labels': []
                    }

                    if len(pred_boxes) > 0:
                        for box, label in zip(pred_boxes, pred_labels):
                            image_pred[image_id]['boxes'].append(box.numpy())
                            image_pred[image_id]['labels'].append(label.item())

                    # Store ground truths for each image
                    image_gt[image_id] = {
                        'boxes': [],
                        'labels': []
                    }

                    if len(gt_boxes) > 0:
                        for box, label in zip(gt_boxes, gt_labels):
                            image_gt[image_id]['boxes'].append(box.numpy())
                            image_gt[image_id]['labels'].append(label.item())

        score = (
            score_sum / len(all_image_ids)
            if len(all_image_ids) > 0
            else 0.0
        )
        mAP = calculate_mAP(all_predictions, all_targets, self.iou_thresholds)

        # Calculate accuracy for each image
        correct_count = 0
        total_count = len(all_image_ids)

        # Process GT and predictions separately for each image
        gt_numbers = {}
        pred_numbers = {}

        # Process ground truth data
        for image_id in all_image_ids:
            gt_data = image_gt.get(image_id, {'boxes': [], 'labels': []})

            if len(gt_data['boxes']) > 0:
                gt_boxes = np.array(gt_data['boxes'])
                gt_labels = np.array(gt_data['labels'])

                # Sort ground truth boxes by x coordinate
                sorted_indices = np.argsort(gt_boxes[:, 0])
                gt_boxes = gt_boxes[sorted_indices]
                gt_labels = gt_labels[sorted_indices]
                gt_number = ''.join(map(str, gt_labels - 1))
                gt_numbers[image_id] = gt_number
            else:
                gt_numbers[image_id] = '-1'

        # Process prediction data
        for image_id in all_image_ids:
            pred_data = image_pred.get(image_id, {'boxes': [], 'labels': []})

            if len(pred_data['boxes']) > 0:
                pred_boxes = np.array(pred_data['boxes'])
                pred_labels = np.array(pred_data['labels'])

                # Sort predicted boxes by x coordinate
                sorted_indices = np.argsort(pred_boxes[:, 0])
                pred_boxes = pred_boxes[sorted_indices]
                pred_labels = pred_labels[sorted_indices]
                pred_number = ''.join(map(str, pred_labels - 1))
                pred_numbers[image_id] = pred_number
            else:
                pred_numbers[image_id] = '-1'

        # Compare GT and predictions for all images
        for image_id in all_image_ids:
            gt_number = gt_numbers[image_id]
            pred_number = pred_numbers[image_id]

            if gt_number == pred_number:
                correct_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0.0

        self.writer.add_scalar('Validation/Average Score', score, epoch)
        self.writer.add_scalar('Validation/mAP', mAP, epoch)
        self.writer.add_scalar('Validation/Accuracy', accuracy, epoch)

        metrics = {
            'score': score,
            'mAP': mAP,
            'accuracy': accuracy
        }

        return metrics

    def predict(self, data_loader=None, score_threshold=0.5):
        """
        Make predictions on a dataset

        Args:
            data_loader: DataLoader to use (default: self.test_loader)
            score_threshold: Confidence threshold for predictions

        Returns:
            List of predictions in COCO format
        """
        if data_loader is None:
            data_loader = self.test_loader

        self.model.eval()
        results = []

        with torch.no_grad():
            for images, image_ids in tqdm(data_loader):
                images = list(image.to(self.device) for image in images)

                # Forward pass
                outputs = self.model(images)

                # Process each image prediction
                for i, output in enumerate(outputs):
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()

                    # Filter predictions by score threshold
                    keep = scores >= score_threshold
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]

                    # Convert boxes: [x1, y1, x2, y2] to [x, y, width, height]
                    coco_boxes = []
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1
                        coco_boxes.append([x1, y1, width, height])

                    # Add predictions to results
                    for box, score, label in zip(coco_boxes, scores, labels):
                        results.append({
                            'image_id': int(image_ids[i]),
                            'bbox': [float(c) for c in box],
                            'score': float(score),
                            'category_id': int(label)
                        })

        return results

    def test(self, score_threshold=0.5):
        """
        Test the model and save predictions to a file

        Args:
            output_file: File to save predictions to
            score_threshold: Confidence threshold for predictions

        Returns:
            List of predictions in COCO format
        """
        results = self.predict(score_threshold=score_threshold)

        # Save results to COCO format
        output_zip = self.args.output_zip
        json_path = output_zip.replace('.zip', '.json')
        csv_path = output_zip.replace('.zip', '.csv')

        if not os.path.exists(os.path.dirname(output_zip)):
            os.makedirs(os.path.dirname(output_zip), exist_ok=True)

        with open(json_path, 'w') as f:
            json.dump(results, f)

        print(f'Predictions saved to {json_path}')

        # Generate the final number result
        self.number_predict(json_file=json_path, csv_file=csv_path)
        print(f'Final predictions saved to {csv_path}')

        # Pack to ZIP
        with zipfile.ZipFile(output_zip, 'w') as zf:
            zf.write(json_path, arcname='pred.json')
            zf.write(csv_path, arcname='pred.csv')
        print(f'ZIP file saved to {output_zip}')

        return results

    def train(self, ckpt_dir='ckpt', save_best=True, save_best_map=True):
        """
        Train the model for multiple epochs

        Args:
            num_epochs: Number of epochs to train for
            save_best: Whether to save the best model based on validation loss

        Returns:
            Dictionary with training and validation losses
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mAP': [],
            'val_accuracy': []
        }

        for epoch in range(self.args.epochs):
            # Train
            train_loss = self.train_one_epoch(epoch)
            print(
                f'Epoch {epoch+1}/{self.args.epochs}, '
                f'Train Loss: {train_loss:.4f}'
            )
            history['train_loss'].append(train_loss)

            self.lr_scheduler.step()

            # Validate
            val_metrics = self.evaluate(epoch)
            val_map = val_metrics['mAP']
            val_accuracy = val_metrics['accuracy']

            print(
                f'Epoch {epoch+1}/{self.args.epochs}, '
                f'mAP: {val_map:.4f}, '
                f'Accuracy: {val_accuracy:.4f}'
            )

            history['val_mAP'].append(val_map)
            history['val_accuracy'].append(val_accuracy)

            # Save best model by accuracy
            if save_best and val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'mAP': val_map,
                    'accuracy': val_accuracy,
                }, self.best_model_path)
                print(f'Best model by accuracy saved at epoch {epoch+1}')

            # Save best model by mAP
            if save_best_map and val_map > self.best_map:
                self.best_map = val_map
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'mAP': val_map,
                    'accuracy': val_accuracy,
                }, self.best_map_model_path)
                print(f'Best model by mAP saved at epoch {epoch+1}')

            print('=' * 30)
            print()

        return history

    def number_predict(self, json_file='pred.json', csv_file='pred.csv'):
        """
        Generate the final prediction file in CSV. It combines all the
        predictions in an image and construct a final number.

        Note: If no number is detected, the prediction will be '-1'.
        """
        # Load predictions from JSON file
        with open(json_file, 'r') as f:
            predictions = json.load(f)

        # Sort the json file by image_id
        predictions.sort(key=lambda x: x['image_id'])

        # Sort the boxes from left to right and combine their category_ids
        # into a single string
        final_predictions = {}
        for pred in predictions:
            image_id = pred['image_id']
            if image_id not in final_predictions:
                final_predictions[image_id] = {
                    'boxes': [],
                    'category_ids': []
                }
            final_predictions[image_id]['boxes'].append(pred['bbox'])
            final_predictions[image_id]['category_ids'].append(
                pred['category_id']
            )

        # Sort the boxes from left to right
        for image_id, pred in final_predictions.items():
            boxes = np.array(pred['boxes'])
            category_ids = np.array(pred['category_ids'])
            sorted_indices = np.argsort(boxes[:, 0])
            boxes = boxes[sorted_indices]
            category_ids = category_ids[sorted_indices]
            final_predictions[image_id]['boxes'] = boxes.tolist()
            final_predictions[image_id]['category_ids'] = category_ids.tolist()
            # Convert category_ids to a string
            final_predictions[image_id]['category_ids'] = ''.join(
                map(str, category_ids - 1)
            )

        # Get all image IDs from test directory
        test_image_ids = set()
        if hasattr(self, 'test_dataset'):
            for img_id in [
                int(os.path.splitext(img)[0])
                for img in self.test_dataset.images
            ]:
                test_image_ids.add(img_id)
        else:
            # If test_dataset is not available, get image IDs from predictions
            for pred in predictions:
                test_image_ids.add(pred['image_id'])

        # Save the final predictions to a CSV file
        results = []
        for image_id in test_image_ids:
            if (image_id in final_predictions and
                    final_predictions[image_id]['category_ids']):
                category_ids = final_predictions[image_id]["category_ids"]
                results.append((image_id, category_ids))
            else:
                # If no prediction for this image_id, write -1
                results.append((image_id, '-1'))

        df = pd.DataFrame(results, columns=['image_id', 'pred_label'])
        df.to_csv(csv_file, index=False)

    def load_best_model(self, metric='accuracy'):
        """
        Load the best model from disk

        Args:
            metric: Metric to use for loading the best model
                ('accuracy' or 'map')
        """
        if metric.lower() == 'accuracy':
            path = self.best_model_path
        elif metric.lower() == 'map':
            path = self.best_map_model_path
        else:
            raise ValueError("metric must be 'accuracy' or 'map'")

        self.load_model(path)
        print(f'Loaded best model by {metric} from {path}')

    def load_model(self, path):
        """Load a model from disk"""
        ckpt = torch.load(path, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.model.to(self.device)
        print(f'Model loaded from {path}')
        print(f'Epoch: {ckpt['epoch']}, '
              f'mAP: {ckpt['mAP']:.4f},'
              f'Accuracy: {ckpt['accuracy']:.4f}')

    def close_tensorboard(self):
        """Close the TensorBoard writer"""
        self.writer.close()


def main():
    if args.train:
        model = FasterRCNN_Res101(args, model_name=model_name)
        print(f'Using device: {model.device}')
        print(f'Number of classes: {model.num_classes}')

        # Train the model
        _ = model.train()
    elif args.test:
        model = FasterRCNN_Res101(args, model_name=model_name)

        # Test on specified checkpoint
        if args.ckpt:
            print(f'Testing with {args.ckpt}...')
            model.load_model(args.ckpt)
            model.test()
        else:
            print('Testing with best checkpoint...')
            model.load_best_model('accuracy')
            model.test()
    elif args.show_test:
        with torch.no_grad():
            model = FasterRCNN_Res101(args, model_name=model_name)
            model.load_best_model()

            img, image_id = model.test_dataset[args.image_id]
            img = img.to(model.device).unsqueeze(0)

            model.model.eval()
            output = model.model(img)[0]
            output['boxes'] = output['boxes'].detach().cpu().numpy()
            output['scores'] = output['scores'].detach().cpu().numpy()
            output['labels'] = output['labels'].detach().cpu().numpy()

            keep = output['scores'] >= model.score_threshold
            output['boxes'] = output['boxes'][keep]
            output['scores'] = output['scores'][keep]
            output['labels'] = output['labels'][keep]

            plot_img_bbox(img, output)
    else:
        print('Please specify --train or --test')

    model.close_tensorboard()


if __name__ == '__main__':
    main()
