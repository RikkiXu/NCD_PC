import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from modules.Discover import Discoverer
from utils import unkn_labels as unk_labels
from utils.callbacks import mIoUEvaluatorCallback

SEED = 1234

parser = ArgumentParser()
parser.add_argument("-s", "--split", type=int, help="split", required=True)
parser.add_argument("--dataset", choices=["SemanticKITTI", "SemanticPOSS"], default="SemanticPOSS", type=str,
                    help="dataset")
parser.add_argument("--dataset_config", default=None, type=str, help="dataset config file")
parser.add_argument("--voxel_size", default="0.05", type=float, help="voxel_size")
parser.add_argument("--downsampling", default="60000", type=int, help="number of points per pcd")
parser.add_argument("--batch_size", default=4, type=int, help="batch size")
parser.add_argument("--num_workers", default=8, type=int, help="number of workers")
parser.add_argument("--hungarian_at_each_step", default=True, action="store_true",
                    help="enable hungarian pass at the end of each epoch")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
parser.add_argument("--train_lr", default=0.001, type=float,
                    help="learning rate for newly initialized parts of the pipeline")
parser.add_argument("--finetune_lr", default=1.0e-4, type=float,
                    help="learning rate for already initialized parts of the pipeline")
parser.add_argument("--use_scheduler", default=False, action="store_true",
                    help="use lr scheduler (linear warm-up + cosine_annealing")
parser.add_argument("--warmup_epochs", default=0, type=int, help="warmup epochs")
parser.add_argument("--detach", default=None, type=int, help="warmup epochs")
parser.add_argument("--min_lr", default=1e-5, type=float, help="min learning rate")
parser.add_argument("--momentum_for_optim", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_for_optim", default=1.0e-4, type=float, help="weight decay")
parser.add_argument("--overcluster_factor", default=None, type=int, help="overclustering factor")
parser.add_argument("--num_heads", default=1, type=int, help="number of heads for clustering")
parser.add_argument("--clear_cache_int", default=1, type=int, help="frequency of clear_cache")
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--initial_epsilon_sk", default=0.05, type=float, help="initial epsilon for the Sinkhorn")
parser.add_argument("--final_epsilon_sk", default=0.05, type=float, help="final epsilon for the Sinkhorn")
parser.add_argument("--adapting_epsilon_sk", default=False, action="store_true",
                    help="use a decreasing value of epsilon for Sinkhorn")
parser.add_argument("--queue_start_epoch", default=2, type=int,
                    help="the epoch in which to start to use the queue. -1 to never use the queue")
parser.add_argument("--queue_batches", default=10, type=int, help="umber of batches in the queue")
parser.add_argument("--queue_percentage", default=0.1, type=float,
                    help="percentage of novel points per batch retained in the queue")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--project", default="NCDPC", type=str, help="wandb project")
parser.add_argument("--entity", default="rikkixu", type=str, help="wandb entity")
parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
parser.add_argument("--epochs", type=int, default=10, help="training epochs")
parser.add_argument("--set_deterministic", default=False, action="store_true")
parser.add_argument("--alpha", default=1, type=float, help="parameters for the loss function")
parser.add_argument("--mix_pl", default=False, action="store_true", help="parameters for the loss function")
parser.add_argument("--use_reweight", default=False, action="store_true", help="parameters for the loss function")
parser.add_argument("--gamma", type=float, default=10)
parser.add_argument("--num_outer_iters", type=int, default=100)
parser.add_argument("--lr_w", type=float, default=0.1)
parser.add_argument("--gamma_decrease", type=float, default=0.1)
parser.add_argument("--ak_bound", type=float, default=0.005)
parser.add_argument("--smooth_bound", type=int, default=10)
parser.add_argument("--lam", type=float, default=3)
parser.add_argument("--lam_region", type=float, default=4)
parser.add_argument("--use_imbalanced_region", default=False, action="store_true", help="")
parser.add_argument("--use_gt", default=False, action="store_true", help="")
parser.add_argument("--exp_path", default=None, help="")
parser.add_argument("--dbscan", type=float, default=0.5)


def main(args):

    os.environ["WANDB_API_KEY"] = "4da1b870fbd955fdee5d0ebb0f28e94ebdaae96d"
    os.environ["WANDB_MODE"] = "offline" if args.offline else "online"

    # args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.comment)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.comment)
    try:
        os.makedirs(args.checkpoint_dir,exist_ok=True)
        os.mkdir(args.log_dir)
    except:
        pass
    print(args)
    run_name = "-".join([f"S{args.split}", "discover", args.dataset, args.comment])
    wandb_logger = WandbLogger(
        save_dir=args.log_dir,
        name=run_name,
        project=args.project,
        entity=args.entity,
        offline=args.offline,
    )

    if args.dataset_config is None:
        if args.dataset == "SemanticKITTI":
            args.dataset_config = "config/semkitti_dataset.yaml"
        elif args.dataset == "SemanticPOSS":
            args.dataset_config = "config/semposs_dataset.yaml"
        else:
            raise NameError(f"Dataset {args.dataset} not implemented")

    with open(args.dataset_config, "r") as f:
        dataset_config = yaml.safe_load(f)

    unknown_labels = unk_labels.unknown_labels(
        split=args.split, dataset_config=dataset_config
    )

    number_of_unk = len(unknown_labels)

    label_mapping, label_mapping_inv, unknown_label = unk_labels.label_mapping(
        unknown_labels, dataset_config["learning_map_inv"].keys()
    )

    args.num_classes = len(label_mapping)
    args.num_unlabeled_classes = number_of_unk
    args.num_labeled_classes = args.num_classes - args.num_unlabeled_classes

    mIoU_callback = mIoUEvaluatorCallback()
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_weights_only=True,
        dirpath=args.checkpoint_dir,
        # every_n_epochs=True,
    )
    csv_logger = CSVLogger(save_dir=args.log_dir)

    loggers = [wandb_logger, csv_logger] if wandb_logger is not None else [csv_logger]

    model = Discoverer(label_mapping, label_mapping_inv, unknown_label, **args.__dict__)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=loggers,
        gpus=-1,
        num_sanity_val_steps=0,
        callbacks=[mIoU_callback, checkpoint_callback, lr_monitor],
    )
    trainer.fit(model)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.set_deterministic:
        os.environ["PYTHONHASHSEED"] = str(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True
    main(args)