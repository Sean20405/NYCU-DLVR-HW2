import os
import json
import time
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet101_Weights
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as T

from utils import calculate_mAP


class FasterRCNN_Res101:
    def __init__(self, args, num_classes, lr=5e-5, model_name=None):
        """
        Initialize the Faster R-CNN model for digit recognition.

        Args:
            num_classes: Number of classes (including background)
            batch_size: Batch size for training
            learning_rate: Initial learning rate
        """
        if model_name is None:
            self.model_name = 'FastRCNN' + time.strftime("%m%d-%H%M%S")
        else:
            self.model_name = model_name
        self.args = args
        self.batch_size = args.batch_size
        self.lr = lr
        self.num_classes = num_classes

        print(f'Model Name: {model_name}')

        # Initialize model, optimizer, and scheduler
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = self._get_model()
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10000, T_mult=2
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

        # Customize anchor
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        model = FasterRCNN(
            backbone=backbone,
            rpn_anchor_generator=anchor_generator,
            num_classes=self.num_classes
        )

        return model

    def _get_transform(self, train):
        """Get transformation pipeline"""
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))

        return T.Compose(transforms)

    def train_one_epoch(self, train_loader, epoch):
        """Train the model for one epoch"""
        self.model.train()

        running_loss = 0.0
        running_loss_dict = {}

        for i, (images, targets) in enumerate(tqdm(train_loader)):
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
            self.lr_scheduler.step()

            running_loss += losses.item()

            # Log training loss and lr
            global_step = epoch * len(train_loader) + i
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

        epoch_loss = running_loss / len(train_loader)
        self.writer.add_scalar('Train/Epoch Loss', epoch_loss, epoch)

        return epoch_loss

    def evaluate(self, valid_loader, epoch):
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
            for images, targets in valid_loader:
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

    def test(self, test_loader, score_threshold=0.5):
        """
        Test the model and save predictions to a file

        Args:
            output_file: File to save predictions to
            score_threshold: Confidence threshold for predictions

        Returns:
            List of predictions in COCO format
        """
        results = self.predict(test_loader, score_threshold)

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

    def train(self, train_loader, valid_loader, save_best=True,
              save_best_map=True):
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
            train_loss = self.train_one_epoch(train_loader, epoch)
            print(
                f'Epoch {epoch+1}/{self.args.epochs}, '
                f'Train Loss: {train_loss:.4f}'
            )
            history['train_loss'].append(train_loss)

            # Validate
            val_metrics = self.evaluate(valid_loader, epoch)
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
        test_dir = os.path.join('data', 'test')
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_id = int(file.split('.')[0])
                    test_image_ids.add(image_id)

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
