#!/usr/bin/env python3
import json
from pathlib import Path
from typing import List, Optional
from zipfile import ZipFile

import liblzfse
import numpy as np
import open3d as o3d
import tqdm
from PIL import Image
from quaternion import as_rotation_matrix, quaternion
from torch.utils.data import Dataset

import sys
sys.path.append('..')

# Import default class labels.
from dataloaders.scannet_200_classes import CLASS_LABELS_200

import tqdm
import einops

import logging
import os
import pprint
import random
from typing import Dict, Union

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import wandb
import sys
sys.path.append('..')

from dataloaders import (
    R3DSemanticDataset,
    DeticDenseLabelledDataset4HSR,
    ClassificationExtractor,
)
from misc import ImplicitDataparallel
from grid_hash_model import GridCLIPModel
from torch.utils.data import Dataset, DataLoader, Subset
from dataloaders import DeticDenseLabelledDataset4HSR

import clip
from sentence_transformers import SentenceTransformer
import time
import glob

# Set up the constants

SAVE_DIRECTORY = "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/clip_implicit_model"
DEVICE = "cuda"
IMAGE_TO_LABEL_CLIP_LOSS_SCALE = 1.0
LABEL_TO_IMAGE_LOSS_SCALE = 1.0
EXP_DECAY_COEFF = 0.5
SAVE_EVERY = 5
METRICS = {
    "accuracy": torchmetrics.Accuracy,
}

BATCH_SIZE = 11000
NUM_WORKERS = 10

CLIP_MODEL_NAME = "ViT-B/32"
SBERT_MODEL_NAME = "all-mpnet-base-v2"

# TODO: Replace with your own path.
DATA_PATH = '/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/clip_fields_add_table.r3d'

CUSTOM_LABELS = [
    "dining table",
    "shelf",
    "sofa",
    "low table",
    "door",
    "fan",
    "cushion",
    "chair",
    "bottle",
    "mattress",
    "plate",
    "bowl",
    "mug",
    "orange",
    "apple",
    "banana",
    "strawberry",
    "spoon",
    "knife",
    "fork",
    "window",
    "blind",
    "tv",
    "remote controller",
    "side table",
    "snack"
    "chips can",
    "cracker box",
    "chocolate",
    "cereal box",
]

TAG = "gpsr"
class ReplaceDataset():
# "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/init"のデータのうちオドメトリが/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect"の各データに対応するものとそれぞれ置き換える
    def replace_dataset():
        if not os.path.exists("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect"):
            return False
        start_time = time.time()
        # "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/"以下に含まれるデータの数を数える
        collect_num = len(glob.glob("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/image*.npy"))
        # "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/"以下に含まれるデータの数を数える
        init_num = len(glob.glob("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/image*.npy"))
        
        exchange_idices = []
        for idx in range(min(collect_num, int(init_num/5))):
            # "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/"のodomに最も近い"/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/"のodomを探す

            # "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/"のodomを取得
            collect_odom = np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/odom"+str(idx).zfill(3)+".npy")

            # "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/"のodomと"/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/"のodomの差を計算
            diff = [np.linalg.norm(collect_odom - np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/odom"+str(i).zfill(3)+".npy"), axis=1) for i in range(init_num)]
            # "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/"のodomに最も近い"/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/"のodomのインデックスを取得
            # diff を値が小さい順に並び替える
            sorted_diff = np.argsort(diff)
            for exchange_idx in sorted_diff:
                if exchange_idx not in exchange_idices:
                    exchange_idices.append(exchange_idx)
                    break

        for idx in range(len(exchange_idices)):

            image = np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/image"+str(idx).zfill(3)+".npy")
            depth = np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/depth"+str(idx).zfill(3)+".npy")
            world = np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/world"+str(idx).zfill(3)+".npy")
            odom = np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/odom"+str(idx).zfill(3)+".npy")

            np.save("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/image"+str(exchange_idices[idx]).zfill(3)+".npy", image)
            np.save("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/depth"+str(exchange_idices[idx]).zfill(3)+".npy", depth)
            np.save("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/world"+str(exchange_idices[idx]).zfill(3)+".npy", world)
            np.save("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/odom"+str(exchange_idices[idx]).zfill(3)+".npy", odom)

        print("time for replace: ", time.time() - start_time)
        return True

class MyDataset(Dataset):
    
    def __init__(self, custom_classes: Optional[List[str]] = CLASS_LABELS_200):
        self.image_lis, self.depth_lis, self.world_lis, self.conf_lis = [], [], [], []
        num = len(glob.glob("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/"+ TAG +"/image*.npy"))
        self.num = num
        image = np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/"+ TAG +'/image000.npy')
        self.image_size = (image.shape[1], image.shape[0])
        if custom_classes:
            self._classes = custom_classes
        else:
            self._classes = CLASS_LABELS_200
            
        self._id_to_name = {i: x for (i, x) in enumerate(self._classes)}
        
        for i in range(self.num):
            num = '000' + str(i)
            num = num[-3:]
            image = np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/"+ TAG +'/image'+num+'.npy').astype(np.uint8)
            depth = (np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/"+ TAG +'/depth'+num+'.npy')*0.001).astype(np.float32)
            mask = ~np.isnan(depth) #& (depth < 3.0)
            self.image_lis.append(image)
            self.depth_lis.append(depth)
            self.conf_lis.append(np.ones_like(depth)*2)

            #depth = depth[mask]
            # depth = cv2.convertScaleAbs(np.load('../data/depth'+num+'.npy')).astype(np.uint8)
            world = np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/"+ TAG +'/world'+num+'.npy').reshape(-1, 3).astype(np.float64)
            self.world_lis.append(world)

    def __len__(self):
        return self.num
    
    def __getitem__(self, idx):
        result = {
            "xyz_position": self.world_lis[idx],
            "rgb": self.image_lis[idx],
            "depth": self.depth_lis[idx],
            "conf": self.conf_lis[idx],
        }
        
        return result


class ClipFieldsTrainer:
    def __init__(self, trained_model_path:str=None) -> None:
        self.trained_model_path = trained_model_path
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        self.sentence_model = SentenceTransformer("all-mpnet-base-v2")
        print("Training model Initialized.")

    def make_dataset(self) -> None:
        start_time = time.time()
        ReplaceDataset().replace_dataset()
        dataset = MyDataset()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)

        for idx, data_dict in tqdm.tqdm(enumerate(dataloader), total=len(dataset), desc="Calculating Detic features"):
            rgb = einops.rearrange(data_dict["rgb"][..., :3], "b h w c -> b c h w")
            xyz = data_dict["xyz_position"]

        labelled_dataset = DeticDenseLabelledDataset4HSR(
            dataset, 
            self.clip_model,
            self.sentence_model,
            use_extra_classes=False, 
            exclude_gt_images=False, 
            use_lseg=False, 
            subsample_prob=0.01, 
            visualize_results=True, 
            detic_threshold=0.4,
            visualization_path="detic_labelled_results_hsr_test",
            # visualization_path="detic_labelled_results_living_add-table",
        )

        torch.save(labelled_dataset, "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/detic_labeled_dataset_hsr.pt")

        # Load the data and create the dataloader created in the previous tutorial notebook

        training_data = torch.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/detic_labeled_dataset_hsr.pt")
        # training_data = torch.load("../detic_labeled_dataset_living_add-table.pt")
        max_coords, _ = training_data._label_xyz.max(dim=0)
        min_coords, _ = training_data._label_xyz.min(dim=0)

        # Set up the model

        if self.trained_model_path is None:
            self.label_model = GridCLIPModel(
                image_rep_size=training_data[0]["clip_image_vector"].shape[-1],
                text_rep_size=training_data[0]["clip_vector"].shape[-1],
                mlp_depth=1,
                mlp_width=600,
                log2_hashmap_size=20,
                num_levels=18,
                level_dim=8,
                per_level_scale=2,
                max_coords=max_coords,
                min_coords=min_coords,
                use_trained_model=False,
                # use_model_weight_path="../clip_implicit_model/implicit_scene_label_model_latest_living.pt",
            ).to(DEVICE)
        else:
            self.label_model = GridCLIPModel(
                image_rep_size=training_data[0]["clip_image_vector"].shape[-1],
                text_rep_size=training_data[0]["clip_vector"].shape[-1],
                mlp_depth=1,
                mlp_width=600,
                log2_hashmap_size=20,
                num_levels=18,
                level_dim=8,
                per_level_scale=2,
                max_coords=max_coords,
                min_coords=min_coords,
                use_trained_model=True,
                use_model_weight_path=self.trained_model_path,
            ).to(DEVICE)

        # label_model.load_state_dict(torch.load("../clip_implicit_model/implicit_scene_label_model_latest_living.pt"))

        self.train_classifier = ClassificationExtractor(
            clip_model_name=CLIP_MODEL_NAME,
            sentence_model_name=SBERT_MODEL_NAME,
            class_names=training_data._all_classes,
            device=DEVICE,
        )

        # Set up our metrics on this dataset.
        self.train_metric_calculators = {}
        train_class_count = {"semantic": self.train_classifier.total_label_classes}
        average_style = ["micro", "macro", "weighted"]
        for classes, counts in train_class_count.items():
            self.train_metric_calculators[classes] = {}
            for metric_name, metric_cls in METRICS.items():
                for avg in average_style:
                    if "accuracy" in metric_name:
                        new_metric = metric_cls(
                            num_classes=counts, average=avg, multiclass=True
                        ).to(DEVICE)
                        self.train_metric_calculators[classes][
                            f"{classes}_{metric_name}_{avg}"
                        ] = new_metric

        # No dataparallel for now
        batch_multiplier = 1

        self.clip_train_loader = DataLoader(
            training_data,
            batch_size=batch_multiplier * BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
        logging.debug(f"Total train dataset sizes: {len(training_data)}")

        # Set up optimizer

        self.optim = torch.optim.Adam(
            self.label_model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.003,
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print("Time taken to make dataset: ", time.time() - start_time)

    @torch.no_grad()
    def zero_shot_eval(self,
        classifier: ClassificationExtractor, 
        predicted_label_latents: torch.Tensor, 
        predicted_image_latents: torch.Tensor, 
        language_label_index: torch.Tensor, 
        metric_calculators: Dict[str, Dict[str, torchmetrics.Metric]]
    ):
        """Evaluate the model on the zero-shot classification task."""
        class_probs = classifier.calculate_classifications(
            model_text_features=predicted_label_latents,
            model_image_features=predicted_image_latents,
        )
        # Now figure out semantic accuracy and loss.
        # Semseg mask is necessary for the boundary case where all the points in the batch are "unlabeled"
        semseg_mask = torch.logical_and(
            language_label_index != -1,
            language_label_index < classifier.total_label_classes,
        ).squeeze(-1)
        if not torch.any(semseg_mask):
            classification_loss = torch.zeros_like(semseg_mask).mean(dim=-1)
        else:
            # Figure out the right classes.
            masked_class_prob = class_probs[semseg_mask]
            masked_labels = language_label_index[semseg_mask].squeeze(-1).long()
            classification_loss = F.cross_entropy(
                torch.log(masked_class_prob),
                masked_labels,
            )
            if metric_calculators.get("semantic"):
                for _, calculators in metric_calculators["semantic"].items():
                    _ = calculators(masked_class_prob, masked_labels)
        return classification_loss

    def train(self,
        clip_train_loader: DataLoader,
        labelling_model: Union[GridCLIPModel, ImplicitDataparallel],
        optim: torch.optim.Optimizer,
        epoch: int,
        classifier: ClassificationExtractor,
        device: Union[str, torch.device] = DEVICE,
        exp_decay_coeff: float = EXP_DECAY_COEFF,
        image_to_label_loss_ratio: float = IMAGE_TO_LABEL_CLIP_LOSS_SCALE,
        label_to_image_loss_ratio: float = LABEL_TO_IMAGE_LOSS_SCALE,
        disable_tqdm: bool = False,
        metric_calculators: Dict[str, Dict[str, torchmetrics.Metric]] = {},
    ):
        """
        Train the model for one epoch.
        """
        total_loss = 0
        label_loss = 0
        image_loss = 0
        classification_loss = 0
        total_samples = 0
        total_classification_loss = 0
        labelling_model.train()
        total = len(clip_train_loader)
        for clip_data_dict in tqdm.tqdm(
            clip_train_loader,
            total=total,
            disable=disable_tqdm,
            desc=f"Training epoch {epoch}",
        ):
            xyzs = clip_data_dict["xyz"].to(device)
            clip_labels = clip_data_dict["clip_vector"].to(device)
            clip_image_labels = clip_data_dict["clip_image_vector"].to(device)
            image_weights = torch.exp(-exp_decay_coeff * clip_data_dict["distance"]).to(
                device
            )
            label_weights = clip_data_dict["semantic_weight"].to(device)
            image_label_index: torch.Tensor = (
                clip_data_dict["img_idx"].to(device).reshape(-1, 1)
            )
            language_label_index: torch.Tensor = (
                clip_data_dict["label"].to(device).reshape(-1, 1)
            )

            (predicted_label_latents, predicted_image_latents) = labelling_model(xyzs)
            # Calculate the loss from the image to label side.
            batch_size = len(image_label_index)
            image_label_mask: torch.Tensor = (
                image_label_index != image_label_index.t()
            ).float() + torch.eye(batch_size, device=device)
            language_label_mask: torch.Tensor = (
                language_label_index != language_label_index.t()
            ).float() + torch.eye(batch_size, device=device)

            # For logging purposes, keep track of negative samples per point.
            image_label_mask.requires_grad = False
            language_label_mask.requires_grad = False
            contrastive_loss_labels = labelling_model.compute_loss(
                predicted_label_latents,
                clip_labels,
                label_mask=language_label_mask,
                weights=label_weights,
            )
            contrastive_loss_images = labelling_model.compute_loss(
                predicted_image_latents,
                clip_image_labels,
                label_mask=image_label_mask,
                weights=image_weights,
            )
            del (
                image_label_mask,
                image_label_index,
                language_label_mask,
            )

            # Mostly for evaluation purposes, calculate the classification loss.
            classification_loss = self.zero_shot_eval(
                classifier, predicted_label_latents, predicted_image_latents, language_label_index, metric_calculators
            )

            contrastive_loss = (
                image_to_label_loss_ratio * contrastive_loss_images
                + label_to_image_loss_ratio * contrastive_loss_labels
            )

            optim.zero_grad(set_to_none=True)
            contrastive_loss.backward()
            optim.step()
            # Clip the temperature term for stability
            labelling_model.temperature.data = torch.clamp(
                labelling_model.temperature.data, max=np.log(100.0)
            )
            label_loss += contrastive_loss_labels.detach().cpu().item()
            image_loss += contrastive_loss_images.detach().cpu().item()
            total_classification_loss += classification_loss.detach().cpu().item()
            total_loss += contrastive_loss.detach().cpu().item()
            total_samples += 1

        to_log = {
            "train_avg/contrastive_loss_labels": label_loss / total_samples,
            "train_avg/contrastive_loss_images": image_loss / total_samples,
            "train_avg/semseg_loss": total_classification_loss / total_samples,
            "train_avg/loss_sum": total_loss / total_samples,
            "train_avg/labelling_temp": torch.exp(labelling_model.temperature.data.detach())
            .cpu()
            .item(),
        }
        for metric_dict in metric_calculators.values():
            for metric_name, metric in metric_dict.items():
                try:
                    to_log[f"train_avg/{metric_name}"] = (
                        metric.compute().detach().cpu().item()
                    )
                except RuntimeError as e:
                    to_log[f"train_avg/{metric_name}"] = 0.0
                metric.reset()
        # wandb.log(to_log)
        logging.debug(pprint.pformat(to_log, indent=4, width=1))
        return total_loss

    def save(self,
        labelling_model: Union[ImplicitDataparallel, GridCLIPModel],
        optim: torch.optim.Optimizer,
        epoch: int,
        save_directory: str = SAVE_DIRECTORY,
        saving_dataparallel: bool = False,
    ):
        if saving_dataparallel:
            to_save = labelling_model.module
        else:
            to_save = labelling_model
        state_dict = {
            "model": to_save.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch,
        }
        torch.save(
            state_dict,
            f"{save_directory}/implicit_scene_label_model_latest_hsr.pt",
            # f"{save_directory}/implicit_scene_label_model_latest_living_add-table.pt",
        )
        return 0
    
    def trainIO(self, epoch:int=5):
        start_time = time.time()
        epoch_cnt = 0
        NUM_EPOCHS = epoch - 1

        while epoch_cnt <= NUM_EPOCHS:
            self.train(
                self.clip_train_loader,
                self.label_model,
                self.optim,
                epoch_cnt,
                self.train_classifier,
                metric_calculators=self.train_metric_calculators,
            )
            epoch_cnt += 1
            if epoch_cnt % SAVE_EVERY == 0:
                self.save(self.label_model, self.optim, epoch_cnt)
        print(f"Total training time: {time.time() - start_time}")

if __name__ == "__main__":

    cft = ClipFieldsTrainer()

    start_time = time.time()
    cft.make_dataset()
    cft.trainIO(5)
    print(f"Total time: {time.time() - start_time}")


