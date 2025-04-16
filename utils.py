import argparse
import numpy as np
import torch
from collections import defaultdict
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_args(model_name):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a model without cross validation'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test for a single model'
    )
    parser.add_argument(
        '--show_test',
        action='store_true',
        help='Visualize a single test image (can specify --image_id)'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU ID'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=6,
        help='Number of workers for DataLoader'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs for training'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for training'
    )
    parser.add_argument(
        '--image_id',
        type=int,
        default=1,
        help='Image ID for visualization'
    )
    parser.add_argument(
        '--output_zip',
        type=str,
        default=f'result/{model_name}.zip',
        help='Output ZIP file for test results'
    )

    return parser.parse_args()


def calculate_ap_at_iou(class_preds, class_targets, iou_threshold):
    """
    Calculate Average Precision at a specific IoU threshold for a class

    Args:
        class_preds: Predictions for this class
        class_targets: Targets for this class
        iou_threshold: IoU threshold for considering a detection as correct

    Returns:
        Average Precision at the specified IoU threshold
    """
    # If no predictions or targets, return 0
    if not class_preds or not class_targets:
        return 0.0

    # Count total number of ground truth boxes
    num_gt_boxes = sum(len(target['boxes']) for target in class_targets)

    # If no ground truth, all predictions are false positives
    if num_gt_boxes == 0:
        return 0.0

    # Collect all predictions across images
    all_scores = []
    all_true_positives = []

    # Create an index for each target image
    gt_boxes_by_image = {}
    for target in class_targets:
        img_idx = target['image_idx']
        gt_boxes_by_image[img_idx] = {
            'boxes': target['boxes'],
            'detected': torch.zeros(len(target['boxes']), dtype=torch.bool)
        }

    # Collect all predictions across all images
    all_predictions = []
    for pred in class_preds:
        img_idx = pred['image_idx']
        boxes = pred['boxes']
        scores = pred['scores']

        for box_idx in range(len(boxes)):
            all_predictions.append({
                'box': boxes[box_idx],
                'score': scores[box_idx],
                'img_idx': img_idx
            })

    # Sort all predictions by score (descending)
    all_predictions.sort(key=lambda x: x['score'], reverse=True)

    # Evaluate each prediction
    for pred in all_predictions:
        img_idx = pred['img_idx']
        box = pred['box']
        score = pred['score']

        all_scores.append(
            score.item() if isinstance(score, torch.Tensor) else score
        )

        # Check if this image has any ground truth
        if (img_idx not in gt_boxes_by_image or
                len(gt_boxes_by_image[img_idx]['boxes']) == 0):
            all_true_positives.append(0)  # False positive
            continue

        # Calculate IoU with all ground truth boxes in this image
        gt_boxes = gt_boxes_by_image[img_idx]['boxes']
        if len(gt_boxes) == 0:
            all_true_positives.append(0)  # False positive
            continue

        # If we only have one box, reshape to ensure proper dimensions
        if len(box.shape) == 1:
            box = box.unsqueeze(0)

        ious = box_iou(box, gt_boxes)[0]  # [0] to get the first (only) row

        # Find best matching ground truth box
        max_iou, max_idx = torch.max(ious, dim=0)

        # Check if the IoU is above threshold and the GT box hasn't been
        # detected yet
        if (max_iou >= iou_threshold and
                not gt_boxes_by_image[img_idx]['detected'][max_idx]):
            all_true_positives.append(1)  # True positive
            gt_boxes_by_image[img_idx]['detected'][max_idx] = True
        else:
            all_true_positives.append(0)  # False positive

    # Convert to numpy arrays for easier calculation
    if not all_scores:
        return 0.0  # No predictions

    scores = np.array(all_scores)
    true_positives = np.array(all_true_positives)

    # Sort by score (descending)
    sorted_indices = np.argsort(-scores)
    true_positives = true_positives[sorted_indices]

    # Compute cumulative true positives
    cum_true_positives = np.cumsum(true_positives)

    # Compute precision and recall
    precision = cum_true_positives / np.arange(1, len(cum_true_positives) + 1)
    recall = cum_true_positives / num_gt_boxes

    # Add sentinel values for calculation
    precision = np.concatenate(([1.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Compute AP as the area under the precision-recall curve
    # Find points where recall changes
    recall_changes = np.where(recall[1:] != recall[:-1])[0] + 1

    # Calculate AP using the rectangle rule
    ap = np.sum(
        (recall[recall_changes] - recall[recall_changes - 1])
        * precision[recall_changes]
    )

    return ap


def calculate_mAP(all_predictions, all_targets, iou_thresholds):
    """
    Calculate mean Average Precision (mAP) for object detection

    Args:
        all_predictions: List of prediction dictionaries
            (each containing 'boxes', 'scores', 'labels')
        all_targets: List of target dictionaries
            (each containing 'boxes', 'labels')
        iou_thresholds: List of IoU thresholds to evaluate

    Returns:
        Mean Average Precision (mAP) across all classes and IoU thresholds
    """
    # Check if predictions or targets are empty
    if not all_predictions or not all_targets:
        return 0.0

    # Organize predictions and targets by class
    class_predictions = defaultdict(list)
    class_targets = defaultdict(list)

    for i, (preds, targets) in enumerate(zip(all_predictions, all_targets)):
        pred_boxes = preds['boxes']
        pred_scores = preds['scores']
        pred_labels = preds['labels']

        gt_boxes = targets['boxes']
        gt_labels = targets['labels']

        # Skip if no predictions or targets
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            continue

        # Get unique labels from both predictions and targets
        unique_labels = torch.unique(
            torch.cat([pred_labels, gt_labels])
            if len(pred_labels) > 0 and len(gt_labels) > 0
            else (pred_labels if len(pred_labels) > 0 else gt_labels)
        )

        # Group by class label
        for label in unique_labels:
            label_item = label.item()

            # Skip background class if present
            if label_item == 0:
                continue

            # Get predictions for this class
            class_mask = pred_labels == label
            class_predictions[label_item].append({
                'boxes': (
                    pred_boxes[class_mask]
                    if len(pred_boxes) > 0 else torch.empty((0, 4))
                ),
                'scores': (
                    pred_scores[class_mask]
                    if len(pred_scores) > 0 else torch.empty(0)
                ),
                'image_idx': i
            })

            # Get targets for this class
            target_mask = gt_labels == label
            class_targets[label_item].append({
                'boxes': (
                    gt_boxes[target_mask]
                    if len(gt_boxes) > 0 else torch.empty((0, 4))
                ),
                'image_idx': i
            })

    # Calculate AP for each class and IoU threshold
    average_precisions = []

    # Iterate over classes
    for class_id in class_predictions.keys():
        # Calculate AP at different IoU thresholds
        aps_for_class = []

        for iou_threshold in iou_thresholds:
            ap = calculate_ap_at_iou(
                class_predictions[class_id],
                class_targets[class_id],
                iou_threshold
            )
            aps_for_class.append(ap)

        # Average AP across IoU thresholds (AP@[.5:.95])
        class_ap = np.mean(aps_for_class) if aps_for_class else 0.0
        average_precisions.append(class_ap)

        print(f"Class {class_id}: {class_ap}")

    # Calculate mAP (mean of APs across all classes)
    mAP = np.mean(average_precisions) if average_precisions else 0.0
    print(f"Overall mAP@[.5:.95]: {mAP:.4f}")

    return mAP


def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)

    img = img.cpu().numpy()

    # Remove batch dimension if present
    if len(img.shape) == 4 and img.shape[0] == 1:  # Shape: (1, C, H, W)
        img = img.squeeze(0)  # Remove batch dimension

    if img.shape[0] == 3:  # Check if the image has 3 channels
        img = img.transpose(1, 2, 0)

    a.imshow(img)
    print(target)
    for idx, (box, label) in enumerate(zip(target['boxes'], target['labels'])):
        x, y, width, height = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)

        # Add the index and label text near the bounding box
        a.text(
            x,
            y - 5,
            f"idx: {idx}, label: {label}",
            color='blue',
            fontsize=8,
        )

    plt.savefig('img_bbox.png')
