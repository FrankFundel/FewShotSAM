import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from utils import SA1B_Dataset
import tqdm
from torch.nn.functional import threshold, normalize

def create_dataloaders(folder_paths, transform, batch_size=32):
    dataloaders = {}

    for folder_path in folder_paths:
        dataset = SA1B_Dataset(root=folder_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataset_name = folder_path.split('/')[-1]  # Extract dataset name from the folder path
        dataloaders[dataset_name] = dataloader

    return dataloaders

def calculate_iou(pred_masks, true_masks):
    intersection = (pred_masks & true_masks).float().sum((1, 2))
    union = (pred_masks | true_masks).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def evaluate_model(model, dataloader):
    model.eval()
    total_iou = 0.0
    total_samples = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for images, masks in tqdm.tqdm(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            outputs = model(images) # , point prompt, conditioning
            pred_masks = torch.argmax(outputs, dim=1)

        total_iou += calculate_iou(pred_masks, masks)
        total_samples += images.size(0)

    average_iou = total_iou / total_samples
    return average_iou

def test_model(model, dataloaders):
    iou_scores = []
    dataset_names = []

    for dataset_name, dataloader in dataloaders.items():
        print("Testing", dataset_name)
        average_iou = evaluate_model(model, dataloader)
        iou_scores.append(average_iou)
        dataset_names.append(dataset_name)

    return iou_scores, dataset_names

def plot_iou_scores(iou_scores, dataset_names, save_path=None):
    num_datasets = len(dataset_names)
    y_pos = np.arange(num_datasets)

    plt.barh(y_pos, iou_scores, align='center', alpha=0.5)
    plt.yticks(y_pos, dataset_names)
    plt.xlabel('IoU Score')
    plt.title('Model Evaluation')

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()  # Display the plot on the screen

# Usage
from segment_anything import SamPredictor, sam_model_registry

class SAM_Baseline(torch.nn.Module):
    def __init__(self, input_size):
        self.sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
        self.input_size = input_size

    def forward(self, x, p, original_image_size):
        #x = sam_model.image_encoder(x)
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
          points=[p],
          boxes=None,
          masks=None
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
          image_embeddings=x,
          image_pe=sam_model.prompt_encoder.get_dense_pe(),
          sparse_prompt_embeddings=sparse_embeddings,
          dense_prompt_embeddings=dense_embeddings,
          multimask_output=False,
        )
        upscaled_masks = sam_model.postprocess_masks(low_res_masks, self.input_size, original_image_size)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
        return binary_mask
    
folder_paths = [
    'FSSAM Datasets/EgoHOS',
    'FSSAM Datasets/GTEA',
    'FSSAM Datasets/LVIS',
    'FSSAM Datasets/NDIS Park',
    'FSSAM Datasets/TrashCan',
    'FSSAM Datasets/ZeroWaste-f',
]
dataloaders = create_dataloaders(folder_paths, batch_size=32)
iou_scores, dataset_names = test_model(model, dataloaders)
plot_iou_scores(iou_scores, dataset_names)