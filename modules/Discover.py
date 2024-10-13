# _*_ coding: utf-8 _*_
"""
Time:     2023/8/16 15:02
Author:   Ruijie Xu
File:     Discoverer_new.py
"""
import os
import sys
from itertools import chain as chain_iterators

import MinkowskiEngine as ME
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.functional import jaccard_index
from tqdm import tqdm
from torch import nn

from torch_scatter import scatter
from models.multiheadminkunet import MultiHeadMinkUnet, MinkUnet
from utils.collation import (
    collation_fn_restricted_dataset,
    collation_fn_restricted_dataset_two_samples,
)
from utils.dataset import dataset_wrapper, get_dataset

from utils.sinkhorn_knopp import SinkhornKnopp, Balanced_sinkhorn_ce, SemiSinkhornKnopp
import numpy as np


def calculate_ASA(superpixel, gt):
    assert superpixel.shape == gt.shape, "Superpixel and ground truth shapes must match."
    unique_labels = np.unique(superpixel)
    total_ASA = 0
    for label in unique_labels:
        mask = superpixel == label
        gt_region = gt[mask]
        current_ASA = np.mean(gt_region == np.argmax(np.bincount(gt_region)))
        total_ASA += current_ASA
    avg_ASA = total_ASA / len(unique_labels)
    return avg_ASA


def split_tensor_by_list(tensor, split_sizes):
    assert sum(split_sizes) == tensor.size(0), "size error"
    splits = []
    start = 0
    for size in split_sizes:
        splits.append(tensor[start:start + size, :])
        start += size
    return splits


def pooling_according_label(pred_array, plabels_array):
    plabels_array = torch.LongTensor(plabels_array).cuda()

    uniques = torch.unique(plabels_array)
    if uniques[0] == -1:
        ps_list = uniques[1:]
    else:
        ps_list = uniques

    max_index = max(ps_list) + 1
    plabels_array[plabels_array == -1] = max_index
    out = scatter(pred_array, plabels_array, dim=0, reduce="mean")
    return out[ps_list, :]


def cluster_acc(y_true, y_pred, num):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)

    t = []
    i = ind[0]
    j = ind[1]
    for k in range(i.shape[0]):
        t.append(w[i[k], j[k]])
    mapping_inv = dict(zip(i[:num], j[:num]))
    return mapping_inv, j, t


def get_weights(split):
    all = [2919206, 573732., 13808048, 1629214., 57947080, 793905., 678539., 79029., 34740372, 141114, 2422128.,
           9526323., 30311148]
    if split == 0:
        ww = torch.zeros((4))
        ww[-1] = all[2]
        ww[0] = all[4]
        ww[1] = all[8]
        ww[2] = all[-1]
    elif split == 1:
        ww = torch.zeros((3))
        ww[-1] = all[0]
        ww[0] = all[-2]
        ww[1] = all[-3]
    elif split == 2:
        ww = torch.zeros((3))
        ww[-1] = all[3]
        ww[0] = all[5]
        ww[1] = all[6]
    elif split == 3:
        ww = torch.zeros((3))
        ww[-1] = all[1]
        ww[0] = all[7]
        ww[1] = all[9]
    ww = 1 / ww
    return ww / ww.sum()


def mse(preds, targets):
    return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)


def cosine_similarity(features, head):
    features = torch.nn.functional.normalize(features, dim=1, p=2)
    head = torch.nn.functional.normalize(head, dim=0, p=2)
    logits = features @ head
    return logits


class Discoverer(pl.LightningModule):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):

        super().__init__()
        self.save_hyperparameters(
            {k: v for (k, v) in kwargs.items() if not callable(v)}
        )
        self.count = 0
        self.region_count = 0

        self.model = MinkUnet(
            num_labeled=self.hparams.num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes
        )

        self.label_mapping = label_mapping
        self.label_mapping_inv = label_mapping_inv
        self.unknown_label = unknown_label

        if self.hparams.pretrained is not None:
            state_dict = torch.load(self.hparams.pretrained)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            print(f'Missing: {missing_keys}', f'Unexpected: {unexpected_keys}')

        self.region_sk = SemiSinkhornKnopp(gamma=self.hparams.gamma)
        self.cos_sk = SemiSinkhornKnopp(gamma=self.hparams.gamma)

        self.loss_per_head = torch.zeros(self.hparams.num_heads, device=self.device)

        # wCE as loss
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.valid_criterion = torch.nn.CrossEntropyLoss()

        # Mapping numeric_label -> word_label
        dataset_config_file = self.hparams.dataset_config
        with open(dataset_config_file, "r") as f:
            dataset_config = yaml.safe_load(f)
        map_inv = dataset_config["learning_map_inv"]
        lab_dict = dataset_config["labels"]
        label_dict = {}
        for new_label, old_label in map_inv.items():
            label_dict[new_label] = lab_dict[old_label]
        self.label_dict = label_dict

        self.train_ps_cosine = []
        self.train_gt = []
        self.asa = []

        return

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.model.parameters(), lr=self.hparams.train_lr,
                                weight_decay=self.hparams.weight_decay_for_optim)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=self.hparams.epochs,
                                                               eta_min=self.hparams.min_lr)

        return [optimizer], [scheduler]

    def train_dataloader(self):

        dataset = get_dataset(self.hparams.dataset)(
            config_file=self.hparams.dataset_config,
            split="train",
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            data_splits=self.hparams.split,
            superpoint=True
        )

        dataset = dataset_wrapper(dataset)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset_two_samples,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

        return dataloader

    def val_dataloader(self):

        dataset = get_dataset(self.hparams.dataset)(
            config_file=self.hparams.dataset_config,
            split="valid",
            voxel_size=self.hparams.voxel_size,
            label_mapping=self.label_mapping,
            data_splits=self.hparams.split
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
        )

        return dataloader

    def on_train_epoch_start(self):
        # Reset best_head tracker
        self.loss_per_head = torch.zeros_like(self.loss_per_head, device=self.device)

    def training_step(self, data, _):
        nlc = self.hparams.num_labeled_classes

        (
            coords,
            feats,
            real_labels,
            selected_idx,
            mapped_labels,
            superpoint_lab,
            selected_region_idx,
            coords1,
            feats1,
            _,
            selected_idx1,
            mapped_labels1,
            superpoint_lab1,
            selected_region_idx1,
            pcd_indexes
        ) = data

        split_sizes = []
        split_sizes1 = []
        batch_num = pcd_indexes.shape[0]
        for i in range(batch_num):
            split_sizes.append(len(superpoint_lab[i]))
            split_sizes1.append(len(superpoint_lab1[i]))

        region_split_sizes = []
        region_split_sizes1 = []
        for i in range(batch_num):
            region_split_sizes.append(len(selected_region_idx[i]) - 1)
            region_split_sizes1.append(len(selected_region_idx1[i]) - 1)

        pcd_masks = []
        pcd_masks1 = []
        for i in range(pcd_indexes.shape[0]):
            pcd_masks.append(coords[:, 0] == i)
            pcd_masks1.append(coords1[:, 0] == i)

        # Forward
        coords = coords.int()
        coords1 = coords1.int()

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords)
        sp_tensor1 = ME.SparseTensor(features=feats1.float(), coordinates=coords1)

        # Clear cache at regular interval
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(sp_tensor)
        out1 = self.model(sp_tensor1)

        logits = torch.cat([out["logits_lab"], out["logits_unlab"]], dim=-1)
        logits1 = torch.cat([out1["logits_lab"], out1["logits_unlab"]], dim=-1)

        mask_lab = mapped_labels != self.unknown_label
        mask_lab1 = mapped_labels1 != self.unknown_label

        # Generate one-hot targets for the base points
        targets_lab = (
            F.one_hot(
                mapped_labels[mask_lab].to(torch.long),
                num_classes=self.hparams.num_labeled_classes,
            )
            .float()
            .to(self.device)
        )
        targets_lab1 = (
            F.one_hot(
                mapped_labels1[mask_lab1].to(torch.long),
                num_classes=self.hparams.num_labeled_classes,
            )
            .float()
            .to(self.device)
        )

        # Generate empty targets for all the points
        targets = torch.zeros_like(logits)
        targets1 = torch.zeros_like(logits1)

        # Insert the one-hot labels
        targets[mask_lab, :nlc] = targets_lab.type_as(targets)
        targets1[mask_lab1, :nlc] = targets_lab1.type_as(targets1)

        all_sp = torch.from_numpy(np.concatenate(superpoint_lab, 0)).int()
        all_sp1 = torch.from_numpy(np.concatenate(superpoint_lab1, 0)).int()
        if targets[~mask_lab].shape[0] != 0 and targets1[~mask_lab1].shape[0] != 0:
            pseudolabel, w_ce, w_reg = self.cos_sk(out["logits_unlab"][~mask_lab])
            targets[~mask_lab, nlc:] = pseudolabel.detach().type_as(targets)

            targets1[~mask_lab1, nlc:] = self.cos_sk(out1["logits_unlab"][~mask_lab1])[0].detach().type_as(targets)

            #print(f"cos_sk: {self.cos_sk.w}, gamma: {self.cos_sk.gamma}")
            freq = torch.bincount(torch.argmax(pseudolabel, dim=1)) / pseudolabel.shape[0]
            #print(f"freq: {freq}")

        re_w_reg = 0
        if torch.all(all_sp == -1) or torch.all(all_sp1 == -1) or targets[~mask_lab].shape[0] == 0 or \
                targets1[~mask_lab1].shape[0] == 0 or self.hparams.alpha == 0:
            # Evaluate loss
            loss_cluster = self.loss(
                10 * logits, targets1, selected_idx, selected_idx1, pcd_masks, pcd_masks1
            )
            loss_cluster += self.loss(
                10 * logits1, targets, selected_idx1, selected_idx, pcd_masks1, pcd_masks
            )
            loss = loss_cluster.mean()
            freq_dict = {f"w/freq{i}": item.item() for i, item in enumerate(freq)}
            w_dict = {f"w/w{i}": item.item() for i, item in enumerate(self.cos_sk.w[0])}

            results = {
                "train/loss": loss.detach(),
                "train/loss_cluster": loss_cluster.detach(),
                "gamma": self.cos_sk.gamma
            }
            results.update(freq_dict)
            results.update(w_dict)
        else:
            regin_feat_list = []
            regin_feat_list1 = []
            feat_list = split_tensor_by_list(out["feats"], split_sizes)
            feat1_list = split_tensor_by_list(out1["feats"], split_sizes1)
            for i in range(batch_num):
                if not np.all(superpoint_lab[i] == -1):
                    regin_feat_list.append(pooling_according_label(feat_list[i], superpoint_lab[i]).cuda())
                if not np.all(superpoint_lab1[i] == -1):
                    regin_feat_list1.append(pooling_according_label(feat1_list[i], superpoint_lab1[i]).cuda())

            region_feat = torch.cat(regin_feat_list, dim=0)
            region_feat1 = torch.cat(regin_feat_list1, dim=0)

            region_logits = cosine_similarity(region_feat, self.model.head_unlab.prototypes.kernel.data)
            targets_region, re_w_ce, re_w_reg = self.region_sk(region_logits)
            targets_region = targets_region.detach().type_as(targets)

            region_logits1 = cosine_similarity(region_feat1, self.model.head_unlab.prototypes.kernel.data)
            targets_region1 = self.region_sk(region_logits1)[0].detach().type_as(targets)

            print(f"re_sk: {self.region_sk.w}, re_gamma: {self.region_sk.gamma}")

            region_logits_list = split_tensor_by_list(region_logits, region_split_sizes)
            region_logits1_list = split_tensor_by_list(region_logits1, region_split_sizes1)
            region_targets_list = split_tensor_by_list(targets_region, region_split_sizes)
            region_targets1_list = split_tensor_by_list(targets_region1, region_split_sizes1)

            # Evaluate loss
            loss_cluster = self.loss(
                10 * logits, targets1, selected_idx, selected_idx1, pcd_masks, pcd_masks1
            )
            loss_cluster += self.loss(
                10 * logits1, targets, selected_idx1, selected_idx, pcd_masks1, pcd_masks
            )

            loss_cluster_region = self.region_loss(region_logits_list, region_targets1_list, selected_region_idx,
                                                   selected_region_idx1, batch_num)
            # if not isinstance(loss_cluster_region_item, (float, int)):
            #     loss_cluster_region = loss_cluster_region_item.mean()
            # else:
            #     loss_cluster_region = 0

            loss_cluster_region += self.region_loss(region_logits1_list, region_targets_list, selected_region_idx1,
                                                    selected_region_idx, batch_num)
            # if not isinstance(loss_cluster_region_item1, (float, int)):
            #     loss_cluster_region += loss_cluster_region_item1.mean()

            # loss = loss_cluster.mean()
            # Keep track of the loss for each head
            # if not isinstance(loss_cluster_region , (float, int)):
            loss = loss_cluster + self.hparams.alpha * loss_cluster_region

            w_dict = {f"w/w{i}": item.item() for i, item in enumerate(self.cos_sk.w[0])}
            region_w_dict = {f"w/region_w{i}": item.item() for i, item in enumerate(self.region_sk.w[0])}

            # logging
            results = {
                "train/loss": loss.detach(),
                "train/loss_cluster": loss_cluster.detach(),
                "train/loss_cluster_region": loss_cluster_region.detach(),
                "train/loss_w_ce": w_ce,
                "train/loss_w_reg": w_reg,
                "gamma": self.cos_sk.gamma,
                "train/re_loss_w_ce": re_w_ce,
                "train/re_loss_w_reg": re_w_reg,
                "regamma": self.region_sk.gamma

            }
            results.update(w_dict)
            results.update(region_w_dict)

        # Keep track of the loss for each head
        self.train_ps_cosine.append(torch.max(F.softmax(out["logits_unlab"][~mask_lab], dim=1), dim=1)[1])
        self.train_gt.append(real_labels[~mask_lab])
        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        self.kl_loss = w_reg
        #print('w_reg: ', self.kl_loss)
        #print('re_w_reg: ', re_w_reg)


        if re_w_reg <= self.hparams.ak_bound:
            self.region_count = self.region_count + 1
            if self.region_count == self.hparams.smooth_bound:
                print("Update gamma from {} to {}".format(self.region_sk.gamma,
                                                          self.region_sk.gamma * self.hparams.gamma_decrease))

                self.region_sk.gamma = self.region_sk.gamma * self.hparams.gamma_decrease
                self.region_count = 0
        else:
            self.region_count = 0

        if self.kl_loss <= self.hparams.ak_bound:
            self.count = self.count + 1
            if self.count == self.hparams.smooth_bound:
                print("Update gamma from {} to {}".format(self.cos_sk.gamma,
                                                          self.cos_sk.gamma * self.hparams.gamma_decrease))

                self.cos_sk.gamma = self.cos_sk.gamma * self.hparams.gamma_decrease
                self.count = 0
        else:
            self.count = 0


        return loss


    # def on_train_epoch_end(self):
    #     ps_cosine = torch.cat(self.train_ps_cosine)
    #     train_gt = torch.cat(self.train_gt)
    #     activate_label = torch.unique(train_gt, sorted=True)

    #     dict_map_cosine, map_cosine, res_cosine = cluster_acc(train_gt.cpu().numpy(), ps_cosine.cpu().numpy(),
    #                                                           self.hparams.num_unlabeled_classes)
    #     map_cosine = map_cosine[:self.hparams.num_unlabeled_classes]
    #     res_cosine = res_cosine[:self.hparams.num_unlabeled_classes]
    #     unseen_recall = torch.zeros(self.hparams.num_unlabeled_classes)
    #     unseen_precision = torch.zeros(self.hparams.num_unlabeled_classes)
    #     for i in range(self.hparams.num_unlabeled_classes):
    #         unseen_recall[i] = res_cosine[i] / torch.sum(train_gt == map_cosine[i])
    #         unseen_precision[i] = res_cosine[i] / torch.sum(ps_cosine == i)
    #     sorted_cosine_mapping_inv = dict(
    #         sorted(dict_map_cosine.items(), key=lambda item: item[1])
    #     )
    #     sorter = list(sorted_cosine_mapping_inv.keys())
    #     unseen_recall = unseen_recall[sorter]
    #     unseen_precision = unseen_precision[sorter]

    #     # logging
    #     results = {
    #         f"recall_cosine/{self.label_dict[activate_label[0].item()]}": unseen_recall[0],
    #         f"recall_cosine/{self.label_dict[activate_label[1].item()]}": unseen_recall[1],
    #         f"recall_cosine/{self.label_dict[activate_label[2].item()]}": unseen_recall[2],
    #         f"recall_cosine/{self.label_dict[activate_label[-1].item()]}": unseen_recall[-1],
    #         f"mapping_cosine/{self.label_dict[map_cosine[0].item()]}": 0,
    #         f"mapping_cosine/{self.label_dict[map_cosine[1].item()]}": 1,
    #         f"mapping_cosine/{self.label_dict[map_cosine[2].item()]}": 2,
    #         f"mapping_cosine/{self.label_dict[map_cosine[-1].item()]}": -1,
    #         f"precision_cosine/{self.label_dict[activate_label[0].item()]}": unseen_precision[0],
    #         f"precision_cosine/{self.label_dict[activate_label[1].item()]}": unseen_precision[1],
    #         f"precision_cosine/{self.label_dict[activate_label[2].item()]}": unseen_precision[2],
    #         f"precision_cosine/{self.label_dict[activate_label[-1].item()]}": unseen_precision[-1],
    #     }

    #     self.train_ps_cosine = []
    #     self.train_gt = []
    #     self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)

    #     # set gamma

    def loss(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            idx_logits: torch.Tensor,
            idx_targets: torch.Tensor,
            pcd_mask_logits: torch.Tensor,
            pcd_mask_targets: torch.Tensor,
            mask_lab_logits=None,
            mask_lab_targets=None,
    ):
        """
        Evaluates the loss function of the predicted logits w.r.t. the targets

        :param logits: predicted logits for the first augmentation of the point clouds
        :param targets: targets for the second augmentation of the point clouds
        :param idx_logits: indexes of the selected points in the first augmentation of the point clouds
        :param idx_targets: indexes of the selected points in the second augmentation of the point clouds
        :param pcd_mask_logits: mask to separate the different point clouds in the batch
        :param pcd_mask_targets: mask to separate the different point clouds in the batch
        """
        loss = 0
        for pcd in range(len(pcd_mask_logits)):
            _idx_logits = pcd_mask_logits[pcd]
            _idx_targets = pcd_mask_targets[pcd]
            if mask_lab_logits is not None and mask_lab_targets is not None:
                _idx_logits = _idx_logits & ~mask_lab_logits
                _idx_targets = _idx_targets & ~mask_lab_targets

            pcd_logits = logits[_idx_logits]
            pcd_targets = targets[_idx_targets]

            logit_shape = pcd_logits.shape[0]
            target_shape = pcd_targets.shape[0]

            if logit_shape == 0 or target_shape == 0:
                continue

            mask_logits = torch.isin(
                idx_logits[_idx_logits], idx_targets[_idx_targets]
            )
            mask_targets = torch.isin(
                idx_targets[_idx_targets], idx_logits[_idx_logits]
            )
            pcd_logits = pcd_logits[mask_logits]
            pcd_targets = pcd_targets[mask_targets]
            ####
            perc_to_log = (pcd_logits.shape[0] / logit_shape
                           + pcd_targets.shape[0] / target_shape
                           ) / 2
            # print(perc_to_log)
            self.log("utils/points_in_common", perc_to_log)

            loss += self.criterion(pcd_logits, pcd_targets).mean()

        return loss / len(pcd_mask_logits)

    def region_loss(
            self, region_logits_list, region_targets1_list, selected_region_idx, selected_region_idx1, batch_num
    ):
        loss = 0
        for pcd in range(batch_num):
            pcd_logits = region_logits_list[pcd]  # 每一个batch的superpoint的logits
            pcd_targets = region_targets1_list[pcd]

            logit_shape = pcd_logits.shape[0]
            target_shape = pcd_targets.shape[0]

            if logit_shape == 0 or target_shape == 0:
                continue
            mask_logits = torch.isin(
                torch.Tensor(selected_region_idx[pcd][1:]), torch.Tensor(selected_region_idx1[pcd][1:])
                # 两个view之间相同的superpoint label
            )
            mask_targets = torch.isin(
                torch.Tensor(selected_region_idx1[pcd][1:]), torch.Tensor(selected_region_idx[pcd][1:])
            )
            pcd_logits = pcd_logits[mask_logits]
            pcd_targets = pcd_targets[mask_targets]
            perc_to_log = (pcd_logits.shape[0] / logit_shape
                           + pcd_targets.shape[0] / target_shape
                           ) / 2

            self.log("utils/points_in_common", perc_to_log)
            loss += self.criterion(pcd_logits, pcd_targets).mean()

        return loss / batch_num

    def on_validation_epoch_start(self):
        # Run the hungarian algorithm to map each novel class to the related semantic class
        if (
                self.hparams.hungarian_at_each_step
                or len(self.label_mapping_inv) < self.hparams.num_classes
        ):
            cost_matrix = torch.zeros(
                (
                    self.hparams.num_unlabeled_classes,
                    self.hparams.num_unlabeled_classes,
                ),
                device=self.device,
            )

            dataset = get_dataset(self.hparams.dataset)(
                config_file=self.hparams.dataset_config,
                split="valid",
                voxel_size=self.hparams.voxel_size,
                label_mapping=self.label_mapping,
                data_splits=self.hparams.split
            )

            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
            )

            real_labels_to_be_matched = [
                label
                for label in self.label_mapping
                if self.label_mapping[label] == self.unknown_label
            ]

            with tqdm(
                    total=len(dataloader), desc="Cost matrix build-up", file=sys.stdout
            ) as pbar:
                for step, data in enumerate(dataloader):
                    coords, feats, real_labels, _, mapped_labels, _ = data

                    # Forward
                    coords = coords.int().to(self.device)
                    feats = feats.to(self.device)
                    real_labels = real_labels.to(self.device)

                    sp_tensor = ME.SparseTensor(
                        features=feats.float(), coordinates=coords
                    )

                    # Must clear cache at regular interval
                    if self.global_step % self.hparams.clear_cache_int == 0:
                        torch.cuda.empty_cache()

                    out = self.model(sp_tensor)

                    mask_unknown = mapped_labels == self.unknown_label

                    preds = out["logits_unlab"]
                    preds = torch.argmax(preds[mask_unknown].softmax(1), dim=1)

                    for pseudo_label in range(self.hparams.num_unlabeled_classes):
                        mask_pseudo = preds == pseudo_label
                        for j, real_label in enumerate(real_labels_to_be_matched):
                            mask_real = real_labels[mask_unknown] == real_label
                            cost_matrix[pseudo_label, j] += torch.logical_and(
                                mask_pseudo, mask_real
                            ).sum()

                    pbar.update()

            cost_matrix = cost_matrix / (
                    torch.negative(cost_matrix)
                    + torch.sum(cost_matrix, dim=0)
                    + torch.sum(cost_matrix, dim=1).unsqueeze(1)
                    + 1e-5
            )

            # Hungarian
            cost_matrix = cost_matrix.cpu()
            row_ind, col_ind = linear_sum_assignment(
                cost_matrix=cost_matrix, maximize=True
            )
            label_mapping = {
                row_ind[i] + self.unknown_label: real_labels_to_be_matched[col_ind[i]]
                for i in range(len(row_ind))
            }
            self.label_mapping_inv.update(label_mapping)

        return

    def validation_step(self, data, _):
        coords, feats, real_labels, _, _, _ = data

        # Forward
        coords = coords.int()

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords)

        # Must clear cache at regular interval
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(sp_tensor)
        preds = torch.cat([out["logits_lab"], out["logits_unlab"]], dim=-1)

        sorted_label_mapping_inv = dict(
            sorted(self.label_mapping_inv.items(), key=lambda item: item[1])
        )
        sorter = list(sorted_label_mapping_inv.keys())

        preds = preds[:, sorter]

        loss = self.valid_criterion(preds, real_labels.long())

        gt_labels = real_labels
        avail_labels = torch.unique(gt_labels).long()
        _, pred_labels = torch.max(torch.softmax(preds.detach(), dim=1), dim=1)
        IoU = jaccard_index(gt_labels, pred_labels, reduction="none")
        IoU = IoU[avail_labels]

        self.log("valid/loss", loss, on_epoch=True, sync_dist=True, rank_zero_only=True)
        IoU_to_log = {
            f"valid/IoU/{self.label_dict[int(avail_labels[i])]}": label_IoU
            for i, label_IoU in enumerate(IoU)
        }
        for label, value in IoU_to_log.items():
            print(label, value)
            self.log(label, value, on_epoch=True, sync_dist=True, rank_zero_only=True)

        return loss

    def on_save_checkpoint(self, checkpoint):
        # Maintain info about best head when saving checkpoints
        checkpoint["loss_per_head"] = self.loss_per_head
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        self.loss_per_head = checkpoint.get(
            "loss_per_head",
            torch.zeros(
                checkpoint["hyper_parameters"]["num_heads"], device=self.device
            ),
        )
        return super().on_load_checkpoint(checkpoint)