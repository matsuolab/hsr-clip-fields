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

from dataloaders.scannet_200_classes import CLASS_LABELS_200

class MyDataset(Dataset):
    
    def __init__(self, num, custom_classes: Optional[List[str]] = CLASS_LABELS_200):
        self.image_lis, self.depth_lis, self.world_lis, self.conf_lis = [], [], [], []
        self.num = num
        image = np.load('../data/image000.npy')
        self.image_size = (image.shape[1], image.shape[0])
        if custom_classes:
            self._classes = custom_classes
        else:
            self._classes = CLASS_LABELS_200
            
        self._id_to_name = {i: x for (i, x) in enumerate(self._classes)}
        
        for i in range(self.num+1):
            num = '000' + str(self.num)
            num = num[-3:]
            image = np.load('../data/image'+num+'.npy')
            depth = (np.load('../data/depth'+num+'.npy')*0.001).astype(np.uint8)
            world = np.load('../data/world'+num+'.npy')
            
            self.image_lis.append(image)
            self.depth_lis.append(depth)
            self.world_lis.append(world)
            self.conf_lis.append(np.ones_like(depth))

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
    
CUSTOM_LABELS = [
    "kitchen counter",
    "kitchen cabinet",
    "stove",
    "cabinet",
    "bathroom counter",
    "refrigerator",
    "microwave",
    "oven",
    "fireplace",
    "door",
    "sink",
    "furniture",
    "dish rack",
    "dining table",
    "shelf",
    "bar",
    "dishwasher",
    "toaster oven",
    "toaster",
    "mini fridge",
    "soap dish",
    "coffee maker",
    "table",
    "bowl",
    "rack",
    "bulletin board",
    "water cooler",
    "coffee kettle",
    "lamp",
    "plate",
    "window",
    "dustpan",
    "trash bin",
    "ceiling",
    "doorframe",
    "trash can",
    "basket",
    "wall",
    "bottle",
    "broom",
    "bin",
    "paper",
    "storage container",
    "box",
    "tray",
    "whiteboard",
    "decoration",
    "board",
    "cup",
    "windowsill",
    "potted plant",
    "light",
    "machine",
    "fire extinguisher",
    "bag",
    "paper towel roll",
    "chair",
    "book",
    "fire alarm",
    "blinds",
    "crate",
    "tissue box",
    "towel",
    "paper bag",
    "column",
    "fan",
    "object",
    "range hood",
    "plant",
    "structure",
    "poster",
    "mat",
    "water bottle",
    "power outlet",
    "storage bin",
    "radiator",
    "picture",
    "water pitcher",
    "pillar",
    "light switch",
    "bucket",
    "storage organizer",
    "vent",
    "counter",
    "ceiling light",
    "case of water bottles",
    "pipe",
    "scale",
    "recycling bin",
    "clock",
    "sign",
    "folded chair",
    "power strip",
]

dataset = MyDataset(48)

from dataloaders import DeticDenseLabelledDataset

labelled_dataset = DeticDenseLabelledDataset(
    dataset, 
    use_extra_classes=False, 
    exclude_gt_images=False, 
    use_lseg=False, 
    subsample_prob=0.01, 
    visualize_results=True, 
    detic_threshold=0.4,
    visualization_path="detic_labelled_results",
)