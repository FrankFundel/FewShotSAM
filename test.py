import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import tqdm
import os
import numpy as np
from torchmetrics.classification import BinaryJaccardIndex

from utils import Embedding_Dataset

jaccard = BinaryJaccardIndex()

def is_valid_file(path):
    if path.endswith(('.jpg', '.png')):
        if os.path.exists(f'{path[:-3]}json') and os.path.exists(f'{path[:-3]}pt'):
            return True
    return False

def create_dataloaders(folder_paths, transform=None, target_transform=None, collate_fn=None, batch_size=8, only_test=False):
    dataloaders = {}

    for folder_path in folder_paths:
        dataset = Embedding_Dataset(root=folder_path, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)

        if only_test:
            dataset_size = len(dataset)
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size
            generator = torch.Generator().manual_seed(42)
            _, _, dataset = random_split(dataset, [train_size, val_size, test_size], generator)

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        dataset_name = folder_path.split('/')[-2]  # Extract dataset name from the folder path
        dataloaders[dataset_name] = dataloader

    return dataloaders

def get_iou(target, prediction, confidence):
    max_i = torch.argmax(confidence, dim=1)
    prediction = prediction[range(len(prediction)), max_i].unsqueeze(1)
    return jaccard(target, prediction.cpu()), prediction

def get_oracle_iou(target, prediction):
    B, N, H, W = prediction.shape
    target_flat = target.repeat(1, N, 1, 1).reshape(B * N, H, W)
    prediction_flat = prediction.reshape(B * N, H, W).cpu()
    iou = []
    for t, p in zip(target_flat, prediction_flat):
        iou.append(jaccard(t, p))
    iou = np.array(iou).reshape(B, N)
    max_i = np.argmax(iou, axis=1)
    prediction = prediction[range(len(prediction)), max_i].unsqueeze(1)
    return jaccard(target, prediction.cpu()), prediction

def evaluate_model(model, dataloader, device):
    model.eval()
    total_iou = 0.0
    total_oracle_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader)
        for images, masks, points, embeddings in pbar:
            embeddings = embeddings.to(device)
            points = points.to(device)

            pred_masks, iou_predictions = model(embeddings, points)
            pred_masks = pred_masks > 0
            
            iou, _ = get_iou(masks, pred_masks, iou_predictions)
            oracle_iou, _ = get_oracle_iou(masks, pred_masks)

            total_iou += iou * images.size(0)
            total_oracle_iou += oracle_iou * images.size(0)
            total_samples += images.size(0)
            
            pbar.set_postfix({'IoU': (total_iou / total_samples).item(), 'oracle IoU': (total_oracle_iou / total_samples).item()})

    average_iou = total_iou / total_samples
    average_oracle_iou = total_oracle_iou / total_samples
    return average_iou, average_oracle_iou

def test_model(model, dataloaders, device):
    iou_scores = []
    oracle_iou_scores = []
    dataset_names = []

    for dataset_name, dataloader in dataloaders.items():
        print("Testing", dataset_name)
        average_iou, average_oracle_iou = evaluate_model(model, dataloader, device)
        print("Average IoU", average_iou)
        print("Average Oracle IoU", average_oracle_iou)
        iou_scores.append(average_iou)
        oracle_iou_scores.append(average_oracle_iou)
        dataset_names.append(dataset_name)

    return iou_scores, oracle_iou_scores, dataset_names

def plot_iou_scores(iou_scores, oracle_iou_scores, dataset_names, save_path=None):
    num_datasets = len(dataset_names)
    bar_width = 0.35  # Width of each bar
    y_pos = np.arange(num_datasets)

    plt.barh(y_pos - bar_width/2, iou_scores, bar_width, align='center', alpha=0.5, label='IoU Scores')
    plt.barh(y_pos + bar_width/2, oracle_iou_scores, bar_width, align='center', alpha=0.5, label='Oracle IoU Scores')
    plt.yticks(y_pos, dataset_names)
    plt.xlabel('IoU Score')
    plt.title('Model Evaluation')
    plt.legend()  # Show legend

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()  # Display the plot on the screen