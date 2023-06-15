import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, cycle
from sentence_transformers import SentenceTransformer, util

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import tqdm
import einops

import os
import sys
sys.path.append("../") # add parent directory to path
from dataloaders.real_dataset import DeticDenseLabelledDataset
from grid_hash_model import GridCLIPModel

from misc import MLP
import pandas as pd
import pyntcloud
from pyntcloud import PyntCloud
import clip
from typing import List, Dict
import time

DEVICE = "cuda"

class ClipFields:
    def __init__(self):
        self.model, preprocess = clip.load("ViT-B/32", device=DEVICE)
        self.sentence_model = SentenceTransformer("all-mpnet-base-v2")
        self.training_data = torch.load("./detic_labeled_dataset_living_add-table.pt")
        max_coords, _ = self.training_data._label_xyz.max(dim=0)
        min_coords, _ = self.training_data._label_xyz.min(dim=0)
        print(max_coords)
        self.label_model = GridCLIPModel(
            image_rep_size=self.training_data[0]["clip_image_vector"].shape[-1],
            text_rep_size=self.training_data[0]["clip_vector"].shape[-1],
            mlp_depth=1,
            mlp_width=600,
            log2_hashmap_size=20,
            num_levels=18,
            level_dim=8,
            per_level_scale=2,
            max_coords=max_coords,
            min_coords=min_coords,
        ).to(DEVICE)

        model_weights_path = "./clip_implicit_model/implicit_scene_label_model_latest_living_add-table.pt"
        model_weights = torch.load(model_weights_path, map_location=DEVICE)
        self.label_model.load_state_dict(model_weights["model"])
        print(self.label_model)
        print("Loaded model from", model_weights_path)

        batch_size = 30_000
        self.points_dataloader = DataLoader(
            self.training_data._label_xyz, batch_size=batch_size, num_workers=10,
        )
        print("Created data loader", self.points_dataloader)

        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(self.training_data._label_xyz)
        merged_pcd.colors = o3d.utility.Vector3dVector(self.training_data._label_rgb)
        merged_downpcd = merged_pcd.voxel_down_sample(voxel_size=0.03)
        print("Create pts result")
        pts_result = np.concatenate((np.asarray(merged_downpcd.points), np.asarray(merged_downpcd.colors)), axis=-1)

        print("initialized")

    def calculate_clip_and_st_embeddings_for_queries(self, queries):
        all_clip_queries = clip.tokenize(queries)
        with torch.no_grad():
            all_clip_tokens = self.model.encode_text(all_clip_queries.to(DEVICE)).float()
            all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
            all_st_tokens = torch.from_numpy(self.sentence_model.encode(queries))
            all_st_tokens = F.normalize(all_st_tokens, p=2, dim=-1).to(DEVICE)
        return all_clip_tokens, all_st_tokens

    def find_alignment_over_model(self, label_model, queries, dataloader, visual=False):
        clip_text_tokens, st_text_tokens = self.calculate_clip_and_st_embeddings_for_queries(queries)
        # We give different weights to visual and semantic alignment 
        # for different types of queries.
        if visual:
            vision_weight = 10.0
            text_weight = 1.0
        else:
            vision_weight = 1.0
            text_weight = 10.0
        point_opacity = []
        with torch.no_grad():
            for data in tqdm.tqdm(dataloader, total=len(dataloader)):
                # Find alignmnents with the vectors
                predicted_label_latents, predicted_image_latents = label_model(data.to(DEVICE))
                data_text_tokens = F.normalize(predicted_label_latents, p=2, dim=-1).to(DEVICE)
                data_visual_tokens = F.normalize(predicted_image_latents, p=2, dim=-1).to(DEVICE)
                text_alignment = data_text_tokens @ st_text_tokens.T
                visual_alignment = data_visual_tokens @ clip_text_tokens.T
                total_alignment = (text_weight * text_alignment) + (vision_weight * visual_alignment)
                total_alignment /= (text_weight + vision_weight)
                point_opacity.append(total_alignment)

        point_opacity = torch.cat(point_opacity).T
        print(point_opacity.shape)
        return point_opacity

    def pred(self, queries: List[str], visual:bool=False, use_threshold:bool=True)->Dict[str, List[float]]:
        alignment_q = self.find_alignment_over_model(self.label_model, queries, self.points_dataloader, visual=visual)
        q = alignment_q[0].squeeze()
        # print(q.shape)
        alpha = q.detach().cpu().numpy()

        # os.makedirs("visualized_pointcloud", exist_ok=True)

        max_points = []
        max_points_goto = []
        mean_points = []
        mean_point_goto = []
        # use_threshold = True
        results = []
        for query, q in zip(queries, alignment_q):
            result = {}
            alpha = q.detach().cpu().numpy()
            pts = self.training_data._label_xyz.detach().cpu()

            # We are thresholding the points to get the top 0.01% of points.
            # Subsample if the number of points is too large.
            threshold = torch.quantile(q[::10, ...], 0.9999).cpu().item()

            # Normalize alpha
            a_norm = (alpha - alpha.min()) / (alpha.max() - alpha.min())
            a_norm = torch.as_tensor(a_norm[..., np.newaxis])
            all_colors = torch.cat((a_norm, torch.zeros_like(a_norm), 1-a_norm), dim=1)

            if use_threshold:
                thres = alpha > threshold
                points = self.training_data._label_xyz[thres]
                max_point = pts[torch.argmax(a_norm)]
                max_points.append(max_point)
                # print(f"LOOKAT {query} {max_point.tolist()}")
                colors = all_colors[thres]
                result[query] = max_point.tolist()
                results.append(result)
            else:
                points = self.training_data._label_xyz
                colors = all_colors
                result[query] = pts[torch.argmax(a_norm)].tolist()
                results.append(result)

        return results

if __name__ == '__main__':
    queries = [
    # Literal
    "Stack of tableware",
    "tv",
    "the sofa",  # intentional misspelling
    "set of fruits",
    "drinks",

    # Visual
    # "white ceramic bowl",
    # "red plastic bowl",
    # "red coffee machine",
    # "espresso machine",
    # "blue garbage bin",
    # "potted plant in a black pot",
    # "purple poster",
    # "toaster oven",

    # Semantic
    "I'm hungry.",
    "I'm thirsty.",
    "I'm tired.",
    "I'm hot.",
    "I want someting sweet",
    # "put away my leftovers",
    "fill out water bottle",
    # "",
    # "warm up my lunch",
    ]
    cf = ClipFields()
    start_time = time.time()
    res = cf.pred(queries, visual=False, use_threshold=True)
    print(f"Time taken: {time.time() - start_time}")
    print(res)
