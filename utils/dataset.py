# _*_ coding: utf-8 _*_
"""
Time:     2023/8/1 16:58
Author:   Ruijie Xu
File:     dataset.py
"""
from copy import deepcopy
import os
import random

import MinkowskiEngine as ME
import numpy as np
import torch
import yaml
# from sklearn import cluster
import pickle
from utils.voxelizer import Voxelizer


def get_dataset(name):
    if name == "SemanticKITTI":
        return SemanticKITTIRestrictedDataset
    elif name == "SemanticPOSS":
        return SemanticPOSSRestrictedDataset
    else:
        raise NameError(f'Dataset "{name}" not yet implemented')


class SemanticKITTIRestrictedDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            config_file="config/dataset.yaml",
            split="train",
            voxel_size=0.05,
            downsampling=80000,
            augment=False,
            label_mapping=None,
            data_splits=None,
            superpoint=False
    ):
        self.superpoint = superpoint
        self.data_splits = data_splits
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                #if not os.path.exists(files_path):
                #    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        self.num_files = len(self.filenames)

        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None
        if data_splits == 3:
            self.unseen_id = 15
        else:
            self.unseen_id = 14

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):

        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        features = np.ones((coordinates.shape[0], 1))

        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF



            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
            sup_label = np.zeros_like(labels) - 1

            if self.superpoint:
                fr = open('Your_Region_path', "rb")
                superpoint = pickle.load(fr)
                fr.close()
                if superpoint is not None:
                    temp = sup_label[labels != -1]
                    temp[self.label_mapping_function(labels[labels != -1]) == self.unseen_id] = superpoint
                    sup_label[labels != -1] = temp

            if self.augment:
                labels = labels[selected_idx]
                sup_label = sup_label[selected_idx]

        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            sup_label = sup_label[labelled_idx]
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)

        discrete_coords, unique_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]
        unique_sup_label = sup_label[unique_map]
        selected_region_idx = np.unique(unique_sup_label)

        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            unique_sup_label,
            selected_region_idx,
            t,
        )



class SemanticKITTIRestrictedDatasetCleanSplit(SemanticKITTIRestrictedDataset):
    def __init__(
            self,
            clean_mask,
            config_file="config/dataset.yaml",
            split="train",
            voxel_size=0.05,
            downsampling=80000,
            augment=False,
            label_mapping=None,
    ):
        super().__init__(
            config_file, split, voxel_size, downsampling, augment, label_mapping
        )
        self.filenames = np.array(self.filenames)[clean_mask]
        self.num_files = len(self.filenames)
        for key in self.files.keys():
            self.files[key] = np.array(self.files[key])[clean_mask]


class SemanticPOSSDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            config_file="config/semposs_dataset.yaml",
            split="train",
            voxel_size=0.05,
            downsampling=80000,
            augment=False,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        learning_map = self.config["learning_map"]
        self.learning_map_function = np.vectorize(lambda x: learning_map[x])

        self.num_files = len(self.filenames)

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        features = np.ones((coordinates.shape[0], 1))

        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            labels = self.learning_map_function(labels)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]

        discrete_coords, unique_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        selected_idx = selected_idx[unique_map]

        return discrete_coords, unique_feats, unique_labels, selected_idx, t


class SemanticPOSSRestrictedDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            config_file="config/semposs_dataset.yaml",
            split="train",
            voxel_size=0.05,
            downsampling=80000,
            augment=False,
            label_mapping=None,
            data_splits=None,
            superpoint=False
    ):
        self.superpoint = superpoint
        self.data_splits = data_splits
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        learning_map = self.config["learning_map"]
        self.learning_map_function = np.vectorize(lambda x: learning_map[x])

        self.num_files = len(self.filenames)

        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None
        '''
        t = 0
        pc_filename = self.files["input"][0]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        label_filename = self.files["label"][t]
        labels = np.fromfile(label_filename, dtype=np.int32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF
        labels = self.learning_map_function(labels)
        coordinates = coordinates[labels != -1]
        mapped_labels = self.label_mapping_function(labels[labels != -1])

        from sklearn import cluster
        clustering = cluster.DBSCAN(eps=0.5, min_samples=2).fit(coordinates[mapped_labels==10])
        region_id = clustering.labels_
        import pickle

        with open("coords_o.pkl", "wb") as f:
            pickle.dump(coordinates[mapped_labels==10], f)
        with open("label.pkl", "wb") as f:
            pickle.dump(region_id, f)
        import ipdb;ipdb.set_trace()
        '''
        if data_splits == 0:
            self.unseen_id = 9
        else:
            self.unseen_id = 10

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):

        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        features = np.ones((coordinates.shape[0], 1))

        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF



            labels = self.learning_map_function(labels)


            sup_label = np.zeros_like(labels) - 1
            if self.superpoint:
                fr = open('Your region path', "rb")
                superpoint = pickle.load(fr)
                fr.close()
                if superpoint is not None:
                    temp = sup_label[labels != -1]
                    temp[self.label_mapping_function(labels[labels != -1]) == self.unseen_id] = superpoint
                    sup_label[labels != -1] = temp

            if self.augment:
                labels = labels[selected_idx]
                sup_label = sup_label[selected_idx]


        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():

            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            sup_label = sup_label[labelled_idx]
            selected_idx = selected_idx[labelled_idx]

            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)

        discrete_coords, unique_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]
        unique_sup_label = sup_label[unique_map]
        selected_region_idx = np.unique(unique_sup_label)

        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            unique_sup_label,
            selected_region_idx,
            t,
        )


class SemanticPOSSRestrictedDatasetCleanSplit(SemanticPOSSRestrictedDataset):
    def __init__(
            self,
            clean_mask,
            config_file="config/semposs_dataset.yaml",
            split="train",
            voxel_size=0.05,
            downsampling=80000,
            augment=False,
            label_mapping=None,
    ):
        super().__init__(
            config_file, split, voxel_size, downsampling, augment, label_mapping
        )
        self.filenames = np.array(self.filenames)[clean_mask]
        self.num_files = len(self.filenames)
        for key in self.files.keys():
            self.files[key] = np.array(self.files[key])[clean_mask]


class dataset_wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, t):
        to_ret = self.dataset.__getitem__(t)[:-1] + self.dataset.__getitem__(t)

        return to_ret