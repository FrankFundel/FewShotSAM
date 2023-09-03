import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode

class SamplePoint(object):
    def __init__(self, non_uniform=True, lambd=10.0):
        self.non_uniform = non_uniform
        self.lambd = lambd

    def __call__(self, masks):
        k = np.random.randint(0, len(masks))
        mask = masks[k]
        width = mask.shape[2]

        if self.non_uniform:
            dist_mask = cv2.copyMakeBorder(np.asarray(mask[0], dtype=np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0) # add 0-border
            dist_mask = cv2.distanceTransform(dist_mask, cv2.DIST_L2, 3)[1:-1, 1:-1] # remove 0-border
            dist_mask = cv2.normalize(dist_mask, None, 0, 1., cv2.NORM_MINMAX)
            flat_mask = dist_mask.flatten()
        else:
            flat_mask = np.asarray(mask[0], dtype=np.float32).flatten()

        valid_indices = np.nonzero(flat_mask)[0]
        if valid_indices.size <= 1:
            return mask, torch.Tensor([0, 0])

        if self.non_uniform:
            p = np.exp(flat_mask[valid_indices] * self.lambd)
        else:
            p = flat_mask[valid_indices]
        p /= p.sum()
        random_index = np.random.choice(valid_indices, p=p)
        point = [random_index % width, random_index // width]
        return mask, torch.Tensor(point)

class PILToNumpy(object):
    def __call__(self, pil_img):
        numpy_img = np.array(pil_img)
        return numpy_img
    
class NumpyToTensor(object):
    def __call__(self, numpy_img):
        tensor_img = np.transpose(numpy_img, (2, 0, 1))
        return torch.from_numpy(tensor_img).float()
    
class SAMPreprocess(object):
    def __init__(self, img_size, normalize=True):
        self.img_size = img_size
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.normalize = normalize
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        if self.normalize:
            x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
class SAMPostprocess(object):
    def __init__(self, img_size):
        self.img_size = img_size
        
    def __call__(self, masks: torch.Tensor) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return masks

def collate(batch):
    images, targets = zip(*batch)
    masks, points = zip(*targets)
    return torch.stack(images), torch.stack(masks), torch.stack(points)

def embedding_collate(batch):
    images, targets, embeddings = zip(*batch)
    masks, points = zip(*targets)
    return torch.stack(images), torch.stack(masks), torch.stack(points), torch.stack(embeddings)

def is_valid_file(path):
    if path.endswith(('.jpg', '.png')):
        if os.path.exists(f'{path[:-3]}json') and os.path.exists(f'{path[:-3]}pt'):
            return True
    return False

def get_square_pad(x1, x2, y1, y2, padding):
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    max_wh = torch.max(w, h).item()
    hp = int((max_wh - w) / 2) + padding
    vp = int((max_wh - h) / 2) + padding
    x1 -= hp
    x2 += hp
    y1 -= vp
    y2 += vp
    return x1, x2, y1, y2

def crop_and_pad(image, coords):
    x1, x2, y1, y2 = coords
    _, image_h, image_w = image.size()
    crop_h = y2 - y1
    crop_w = x2 - x1
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - image_w)
    pad_bottom = max(0, y2 - image_h)
    cropped_image = image[:, max(y1, 0):min(y2, image_h), max(x1, 0):min(x2, image_w)]
    padded_image = F.pad(cropped_image, (pad_left, pad_right, pad_top, pad_bottom))
    return padded_image

def create_cutouts(image, masks, size, padding=0, background_transform=None, background_intensity=0):
    transform = Compose([
        Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True),
    ])
    cutouts = torch.empty((len(masks), 3, size, size))
    for i, mask in enumerate(masks):
        horizontal_indicies = torch.where(torch.any(mask, dim=0))[0]
        vertical_indicies = torch.where(torch.any(mask, dim=1))[0]
        if horizontal_indicies.shape[0] and vertical_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            coords = get_square_pad(x1, x2, y1, y2, padding)
            
            if background_transform is not None:
                cutout = image * mask
                cropped = crop_and_pad(cutout, coords)
                
                background = crop_and_pad(image, coords)
                background = background * background_intensity + torch.full_like(background, 0.5) * (1-background_intensity)
                background = background_transform(background)
                
                mask = crop_and_pad(mask.unsqueeze(0), coords).squeeze(0)
                background[:, mask > 0] = cropped[:, mask > 0]
                cutouts[i] = transform(background)
            else:
                cutout = image * mask
                cutout[:, mask==0] = torch.full_like(image, 0.5)[:, mask==0]
                cropped = crop_and_pad(cutout, coords)
                cutouts[i] = transform(cropped)
    return cutouts

def create_cutout_split(image, masks, size, padding=0):
    transform = Compose([
        Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True),
    ])
    images_split = torch.empty((len(masks), 3, size, size))
    masks_split = torch.empty((len(masks), 1, size, size))
    for i, mask in enumerate(masks):
        horizontal_indicies = torch.where(torch.any(mask, dim=0))[0]
        vertical_indicies = torch.where(torch.any(mask, dim=1))[0]
        if horizontal_indicies.shape[0] and vertical_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            coords = get_square_pad(x1, x2, y1, y2, padding)
            img = crop_and_pad(image, coords)
            msk = crop_and_pad(mask.unsqueeze(0), coords)
            images_split[i] = transform(img)
            masks_split[i] = transform(msk)
    return images_split, masks_split
