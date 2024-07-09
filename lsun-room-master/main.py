import argparse
import importlib

import pytorch_lightning as pl
from loguru import logger
from memory_profiler import profile
from trainer import core


def create_dataset(args):
    module = importlib.import_module(f'datasets.{args.dataset}')
    Dataset = getattr(module, {
        'lsunroom': 'LsunRoomDataset',
        'hedau': 'HedauDataset',
    }[args.dataset])

    train_dataset = Dataset('training', folder=args.train_folder, image_size=args.image_size)
    val_dataset = Dataset('validation', folder=args.val_folder, image_size=args.image_size)

    return (
        train_dataset.to_loader(batch_size=args.batch_size, num_workers=args.worker),
        val_dataset.to_loader(batch_size=1, num_workers=args.worker),
    )

def main(args):
    logger.info(args)

    train_loader, val_loader = create_dataset(args)
    if args.phase == 'train':
        model = core.LayoutSeg(
            lr=args.lr, backbone=args.backbone,
            l1_factor=args.l1_factor, l2_factor=args.l2_factor, edge_factor=args.edge_factor
        )
        trainer = pl.Trainer(
            max_epochs=args.epoch,
            resume_from_checkpoint=args.pretrain_path or None,
            accumulate_grad_batches=4,
            checkpoint_callback=False,  # Disable checkpointing if not needed
            log_every_n_steps=50,  # Log every 50 steps
            flush_logs_every_n_steps=1,  # Flush logs every 100 steps
        )
        train_loader, val_loader = create_dataset(args)

        trainer.fit(model, train_loader, val_loader)
    elif args.phase == 'eval':
        model = core.LayoutSeg.load_from_checkpoint(args.pretrain_path, backbone=args.backbone)
        trainer = pl.Trainer(gpus=1, logger=None)
        result = trainer.test(model, val_loader)
        logger.info(f'Validate score on {args.dataset}: {result[0]["score"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Indoor room corner detection')
    parser.add_argument('--name', help='experiment name')
    parser.add_argument('--train_folder', help='where is the training dataset')
    parser.add_argument('--val_folder', help='where is the validation dataset')
    parser.add_argument('--dataset', default='lsunroom', choices=['lsunroom', 'hedau'])
    parser.add_argument('--phase', default='eval', choices=['train', 'eval'])
    parser.add_argument('--worker', default=12, type=int)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)

    # data
    parser.add_argument('--image_size', default=320, type=int)
    parser.add_argument('--use_layout_degradation', action='store_true')

    # network
    parser.add_argument('--arch', default='resnet')
    parser.add_argument('--backbone', default='resnet101')
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--pretrain_path', default='')

    # hyper-parameters
    parser.add_argument('--l1_factor', type=float, default=0.0)
    parser.add_argument('--l2_factor', type=float, default=0.0)
    parser.add_argument('--edge_factor', type=float, default=0.0)
    args = parser.parse_args()

    main(args)
