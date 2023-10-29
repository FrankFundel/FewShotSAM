#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
import monai
import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.nn.parallel import DataParallel
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import threshold, normalize
from torchmetrics.classification import BinaryJaccardIndex

torch.backends.cuda.matmul.allow_tf32 = True
print(torch.__version__)

from segment_anything.utils.transforms import ResizeLongestSide

from datasets import Embedding_Dataset
from utils import SAMPreprocess, PILToNumpy, NumpyToTensor, SamplePoint, embedding_collate, is_valid_file2
from utils import create_cutouts
from models import SAM_Concat, SAM_GSA


# In[2]:

parser = argparse.ArgumentParser(description='Process command line arguments.')

parser.add_argument('--embed_model_type', type=str, choices=['clip', 'dino'],
                    default='clip', help='Embed model type: clip or dino (default: clip)')
parser.add_argument('--guidance_method', type=str, choices=['concat', 'gsa'],
                    default='gsa', help='Guidance method: concat or gsa (default: gsa)')
parser.add_argument('--example_method', type=str, choices=['ground_truth', 'noisy_ground_truth', 'retrieval'],
                    default='ground_truth', help='Example method: ground_truth, noisy_ground_truth, retrieval (default: ground_truth)')
parser.add_argument('--multi_output', action='store_true', help='Enable multi-output')
                    #default=False
parser.add_argument('--visual_prompt_engineering', action='store_true', help='Enable visual prompt engineering')
                    #default=False

# Extra options
parser.add_argument('--epochs', type=int,
                    default=10,help='Number of training epochs (default: 10)')
parser.add_argument('--batch_size', type=int,
                    default=64, help='Batch size for training (default: 64)')
parser.add_argument('--num_workers', type=int,
                    default=35, help='Number of workers for data loading (default: 35)')
parser.add_argument('--lr', type=float,
                    default=1e-5, help='Learning rate for training (default: 1e-5)')
parser.add_argument('--no_log', action='store_true', help='Disable logging with wandb')
                    #default=False


args = parser.parse_args()

# Accessing the values
embed_model_type = args.embed_model_type
guidance_method = args.guidance_method
example_method = args.example_method
multi_output = args.multi_output
visual_prompt_engineering = args.visual_prompt_engineering

no_log = args.no_log

# Your logic or function calls using the arguments go here
print(f"Embed Model Type: {embed_model_type}")
print(f"Guidance Method: {guidance_method}")
print(f"Example Method: {example_method}")
print(f"Multi-Output: {multi_output}")
print(f"Visual Prompt Engineering: {visual_prompt_engineering}")

save_name = 'SAM_single_' + guidance_method + '_' + embed_model_type + '_' + example_method + '.pt'


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


if embed_model_type == 'clip':
    import clip
    class CLIP:
        def __init__(self, device):
            self.model, _ = clip.load("ViT-L/14@336px", device=device)
            self.model.eval()
            self.model.to(device)
        def __call__(self, image):
            with torch.no_grad():
                return self.model.encode_image(image)
    example_dtype = torch.float16
    embed_model = CLIP(device)
    cutout_size = 336
    example_dim = 768
    print("CLIP loaded", cutout_size)
elif embed_model_type == 'dino':
    class DINO:
        def __init__(self, device):
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.model.eval()
            self.model.to(device)
        def __call__(self, image):
            with torch.no_grad():
                return self.model(image)
    example_dtype = torch.float32
    embed_model = DINO(device)
    cutout_size = 336
    example_dim = 1024
    print("DINO loaded", cutout_size)


# In[5]:


if guidance_method == 'concat':
    model = SAM_Concat(example_dim, example_dtype, multi_output)
elif guidance_method == 'gsa':
    model = SAM_GSA(example_dim, multi_output)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model)
    model_attr = model.module
else:
    model_attr = model
model.to(device)

# make sure we only compute gradients for mask decoder
for name, param in model_attr.sam_model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

if guidance_method == 'gsa':
    for name, param in model_attr.sam_model.named_parameters():
        if not "gated_self_attn" in name:
            param.requires_grad_(False)

jaccard = BinaryJaccardIndex().to(device)


# In[6]:


sam_transform = ResizeLongestSide(model_attr.img_size)
target_transform = Compose([
    sam_transform.apply_image_torch, # rescale
    SAMPreprocess(model_attr.img_size, normalize=False), # padding
    SamplePoint(False),
])
transform = Compose([
    PILToNumpy(),
    sam_transform.apply_image, # rescale
    NumpyToTensor(),
    SAMPreprocess(model_attr.img_size) # padding
])


# In[7]:


epochs = args.epochs
batch_size = args.batch_size
num_workers = args.num_workers
lr = args.lr

print(f"Epochs: {epochs}")
print(f"Batch Size: {batch_size}")
print(f"Number of Workers: {num_workers}")
print(f"Learning Rate: {lr}")

folder_path = '/pfs/work7/workspace/scratch/ul_xto11-FSSAM/sa1b'
dataset = Embedding_Dataset(mask_json=False, root=folder_path, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file2)

dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size
generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator)

train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=embedding_collate,
                          num_workers=num_workers, pin_memory=True, shuffle=False)
val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=embedding_collate,
                        num_workers=num_workers, pin_memory=True, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=embedding_collate,
                         num_workers=num_workers, pin_memory=True, shuffle=False)

# In[8]:


optimizer = torch.optim.Adam(model_attr.sam_model.mask_decoder.parameters(), lr=lr, weight_decay=0)
criterion = monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, lambda_focal=20.) # maybe include_background=False


# In[9]:


def get_oracle(target, prediction):
    B, N, H, W = prediction.shape
    target_flat = target.repeat(1, N, 1, 1).reshape(B * N, H, W)
    prediction_flat = prediction.reshape(B * N, H, W)
    iou = []
    for t, p in zip(target_flat, prediction_flat):
        iou.append(jaccard(t, p))
    iou = torch.tensor(iou).reshape(B, N).to(device)
    max_i = torch.argmax(iou, 1)
    return iou, max_i


# In[10]:


from torchvision.transforms import GaussianBlur

class CreateCutouts(object):
    def __init__(self, cutout_size, padding, background_transform, background_intensity):
        self.cutout_size = cutout_size
        self.padding = padding
        self.background_transform = background_transform
        self.background_intensity = background_intensity

    def __call__(self, image, masks):
        #image = (image - pixel_mean) / pixel_std # better performance without normilization
        return create_cutouts(image, masks, self.cutout_size, self.padding, self.background_transform, self.background_intensity)
        
if visual_prompt_engineering:
    background_intensity = .1
    background_transform = Compose([
        GaussianBlur(11, 10)
    ])
else:
    background_intensity = 0
    background_transform = None

create_cutouts_f = CreateCutouts(cutout_size, 30, background_transform, background_intensity)


# In[11]:


def train_epoch(model, epoch, criterion, optimizer, dataloader, device):
    model.train()
    
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    
    running_loss = 0.0
    running_iou = 0.0

    for batch, (images, masks, points, embeddings) in enumerate(tqdm.tqdm(dataloader)):
        # Transfer Data to GPU if available
        embeddings, points = embeddings.to(device), points.to(device)

        # Clear the gradients
        optimizer.zero_grad()
        
        # Get cutout embeddings
        cutouts = [create_cutouts_f(i, m) for i, m in zip(images, masks)]
        cutouts = torch.cat(cutouts).to(device)
        cutout_embeddings = embed_model(cutouts)
        
        # Get examples
        if example_method == 'ground_truth':
            examples = cutout_embeddings.unsqueeze(1) # use groundtruth cutout embedding as example
            # examples = torch.cat([examples, torch.rand((len(images), 10, example_dim), dtype=torch.float16).to(device)], 1) # augment examples with random vectors
        elif example_method == 'noisy_ground_truth':
            pass
            #examples = cutout_embeddings.unsqueeze(1).repeat(1, 5, 1) # use groundtruth cutout embedding as example
        elif example_method == 'retrieval':
            pass

        masks = masks.to(device)
        
        # Forward Pass
        if multi_output:
            outputs, iou_pred = model(embeddings, points, examples)
            iou, oracle_i = get_oracle(masks > 0, outputs > 0)
            oracle_outputs = outputs[range(len(outputs)), oracle_i].unsqueeze(1)
            outputs = outputs[range(len(outputs)), torch.argmax(iou_pred, dim=1)].unsqueeze(1)
            loss = criterion(oracle_outputs, masks) + 0.1 * torch.nn.functional.mse_loss(iou_pred, iou) # Compute Loss
        else:
            outputs = model(embeddings, points, examples)
            loss = criterion(outputs, masks) # Compute Loss

        # Calculate gradients
        loss.backward()

        # Update Weights
        optimizer.step()

        # Calculate Loss
        running_loss += loss.item() * embeddings.size(0)
        running_iou += jaccard(masks > 0, outputs > 0) * embeddings.size(0)

    epoch_loss = running_loss / num_samples
    epoch_iou = running_iou / num_samples

    return epoch_loss, epoch_iou


# In[12]:


def test_epoch(model, epoch, criterion, optimizer, dataloader, device):
    model.eval()
    
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    
    with torch.no_grad():
        running_loss = 0.0
        running_iou = 0.0

        for batch, (images, masks, points, embeddings) in enumerate(tqdm.tqdm(dataloader)):
            # Transfer Data to GPU if available
            embeddings, points = embeddings.to(device), points.to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Get cutout embeddings
            cutouts = [create_cutouts_f(i, m) for i, m in zip(images, masks)]
            cutouts = torch.cat(cutouts).to(device)
            cutout_embeddings = embed_model(cutouts)

            # Get examples
            if example_method == 'ground_truth':
                examples = cutout_embeddings.unsqueeze(1) # use groundtruth cutout embedding as example
                #examples = torch.concat(examples, torch.rand()) # augment examples with random vectors
            elif example_method == 'noisy_ground_truth':
                pass
            elif example_method == 'retrieval':
                pass

            masks = masks.to(device)
        
            # Forward Pass
            if multi_output:
                outputs, iou_pred = model(embeddings, points, examples)
                iou, oracle_i = get_oracle(masks > 0, outputs > 0)
                oracle_outputs = outputs[range(len(outputs)), oracle_i].unsqueeze(1)
                outputs = outputs[range(len(outputs)), torch.argmax(iou_pred, dim=1)].unsqueeze(1)
                loss = criterion(oracle_outputs, masks) + 0.1 * torch.nn.functional.mse_loss(iou_pred, iou) # Compute Loss
            else:
                outputs = model(embeddings, points, examples)
                loss = criterion(outputs, masks) # Compute Loss

            # Calculate Loss
            running_loss += loss.item() * embeddings.size(0)
            running_iou += jaccard(masks > 0, outputs > 0) * embeddings.size(0)
            
        epoch_loss = running_loss / num_samples
        epoch_iou = running_iou / num_samples

    return epoch_loss, epoch_iou


# In[13]:


if not no_log:
    import wandb
    
    wandb_config = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
    }

    wandb.init(project="SAM_single", entity="frankfundel", config=wandb_config)


# In[14]:


min_val_loss = np.inf

for epoch in range(epochs):
    end = time.time()
    print(f"==================== Starting at epoch {epoch} ====================", flush=True)
    
    train_loss, train_iou = train_epoch(model, epoch, criterion, optimizer, train_loader, device)
    print('Training loss: {:.4f} IoU: {:.4f}'.format(train_loss, train_iou), flush=True)
    
    val_loss, val_iou = test_epoch(model, epoch, criterion, optimizer, val_loader, device)
    print('Validation loss: {:.4f} IoU: {:.4f}'.format(val_loss, val_iou), flush=True)
    
    if not no_log:
        wandb.log({
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
        })
    
    if min_val_loss > val_loss:
        print('val_loss decreased, saving model', flush=True)
        min_val_loss = val_loss
        
        # Saving State Dict
        torch.save(model.state_dict(), save_name)


# In[ ]:


# Load after training
model_attr.load_state_dict(torch.load(save_name))


# In[ ]:


test_loss, test_iou = test_epoch(model, 0, criterion, optimizer, test_loader, device)
print('Test loss: {:.4f} IoU: {:.4f}'.format(test_loss, test_iou), flush=True)


# In[ ]:


if not no_log:
    wandb.log({
        "test_loss": test_loss,
        "test_iou": test_iou
    })
    
    wandb.finish()

