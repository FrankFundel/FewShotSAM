import torch
import torchvision
import json
from pycocotools import mask as mask_utils
import numpy as np

class SA1B_Dataset(torchvision.datasets.ImageFolder):
    """A data loader for the SA-1B Dataset from "Segment Anything" (SAM)"""

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.imgs[index] # discard automatic subfolder labels
        sample = self.loader(path)
        masks = json.load(open(f'{path[:-3]}json'))['annotations'] # load json masks
        target = []

        for m in masks:
            # decode masks from COCO RLE format
            mask = mask_utils.decode(m['segmentation'])
            if len(mask.shape) == 3:
                target.append(torch.as_tensor(mask).permute(2, 0, 1))
            else:
                target.append(torch.as_tensor(mask))
        target = torch.stack(target)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.imgs)

class Path_Dataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.imgs[index] # discard automatic subfolder labels
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

    def __len__(self):
        return len(self.imgs)
    
class Embedding_Dataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, masks, embedding).
        """
        path, _ = self.imgs[index] # discard automatic subfolder labels
        sample = self.loader(path)
        masks = json.load(open(f'{path[:-3]}json'))['annotations'] # load json masks
        embedding = torch.load(f'{path[:-3]}pt') # load embedding
        target = []

        for m in masks:
            seg = m['segmentation']
            if isinstance(seg[0], list):  # Polygon format
                mask = mask_utils.decode(mask_utils.frPyObjects(seg, sample.size[1], sample.size[0]))
            elif isinstance(seg[0], dict):  # RLE format
                mask = mask_utils.decode(seg)
            if np.sum(mask) > 0:
                target.append(torch.as_tensor(mask).permute(2, 0, 1))
        
        if len(target) > 0:
            target = torch.concat(target).unsqueeze(1)
        else:
            target = torch.zeros(1, 1, sample.size[1], sample.size[0])
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, embedding

    def __len__(self):
        return len(self.imgs)