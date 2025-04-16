import os
import json

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from model import FasterRCNN_Res101
from dataset import (
    CocoDataset,
    TestDataset
)
from utils import (
    parse_args,
    plot_img_bbox
)

# !!! Remember to change this for each experiment !!!
model_name = 'FasterRCNN_Res101_anchor'

args = parse_args(model_name)

# Set GPU ID
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)


def main():
    # Data directories
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    train_json = os.path.join(data_dir, 'train.json')
    valid_json = os.path.join(data_dir, 'valid.json')

    # Determine number of classes if not provided
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    categories = train_data['categories']
    num_classes = len(categories) + 1  # +1 for background class

    if args.train:
        train_dataset = CocoDataset(
            train_dir, train_json,
            transforms=T.Compose([
                T.ToTensor(),
                T.RandomHorizontalFlip(0.5),
            ])
        )
        valid_dataset = CocoDataset(
            valid_dir, valid_json,
            transforms=T.Compose([
                T.ToTensor(),
                T.RandomHorizontalFlip(0.5),
            ])
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda x: tuple(zip(*x))
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda x: tuple(zip(*x))
        )

        model = FasterRCNN_Res101(args, num_classes, model_name=model_name)

        print(f'Using device: {model.device}')
        print(f'Number of classes: {model.num_classes}')

        # Train the model
        _ = model.train(train_loader, valid_loader)
    elif args.test:
        test_dataset = TestDataset(
            test_dir,
            transforms=T.Compose([T.ToTensor()])
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size*2, shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda x: tuple(zip(*x))
        )

        model = FasterRCNN_Res101(args, num_classes, model_name=model_name)

        # Test on specified checkpoint
        if args.ckpt:
            print(f'Testing with {args.ckpt}...')
            model.load_model(args.ckpt)
            model.test(test_loader)
        else:
            print('Testing with best checkpoint...')
            model.load_best_model('accuracy')
            model.test(test_loader)
    elif args.show_test:
        with torch.no_grad():
            test_dataset = TestDataset(
                test_dir,
                transforms=T.Compose([T.ToTensor()])
            )

            model = FasterRCNN_Res101(args, num_classes, model_name=model_name)
            model.load_best_model()

            img, image_id = test_dataset[args.image_id]
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
