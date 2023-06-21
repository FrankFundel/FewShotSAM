import torch
import torchvision
import torchvision.transforms as transforms
import json
from PIL import Image
from pycocotools import mask as mask_utils
import random
from torch.utils.data import Dataset

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
            target.append(torch.Tensor(mask_utils.decode(m['segmentation'])))
        target = torch.stack(target)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.imgs)

def sample_point(masks):
    k_mask = random.randint(0, len(masks)-1)
    ground_truth_mask = masks[k_mask]
    height, width, _ = ground_truth_mask.shape
    flat_mask = ground_truth_mask.view(-1)
    valid_indices = torch.nonzero(flat_mask).squeeze()
    random_index = random.randint(0, len(valid_indices)-1)
    point = [valid_indices[random_index] // width, valid_indices[random_index] % width]
    return point, ground_truth_mask
