import os
import sys
import json
import numpy as np
from PIL import Image
import base64
import tqdm
from pycocotools import mask as mask_utils

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentation.append(contour)
    return segmentation

def convert_masks_to_coco_json(root_folder, json_path):
    image_folder = os.path.join(root_folder, "images")
    mask_folder = os.path.join(root_folder, "masks")

    # Retrieve a list of all mask image files in the folder
    image_files = [f for f in os.listdir(image_folder)]

    # Create a dictionary for the COCO JSON format
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    for i, image_file in tqdm.tqdm(enumerate(image_files)):
        # Read the image file
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, image_file.replace("jpg", "png"))
        if not os.path.exists(mask_path):
            continue

        # Load the image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert("L")

        # Convert the mask to a numpy array
        mask_array = np.asfortranarray(mask)

        # Encode the binary mask
        encoded_mask = mask_utils.encode(mask_array)
        encoded_mask["counts"] = encoded_mask["counts"].decode("utf-8")

        # Create an annotation entry for the image
        annotation = {
            "id": i + 1,
            "image_id": i + 1,
            "category_id": 0,  # No category/label
            "segmentation": [encoded_mask],
            "iscrowd": 0
        }

        # Add the image and annotation entries to the COCO data
        coco_data["images"].append({
            "id": i + 1,
            "file_name": image_file,
            "height": image.height,
            "width": image.width,
        })

        coco_data["annotations"].append(annotation)

    # Save the COCO JSON file
    with open(json_path, "w") as json_file:
        json.dump(coco_data, json_file)


if __name__ == "__main__":
    # Check if the root folder path argument is provided
    if len(sys.argv) != 2:
        print("Please provide the root folder path as an argument.")
        print("Usage: python convert_masks_to_coco.py /path/to/root/folder")
        sys.exit(1)

    # Retrieve the root folder path from the command-line argument
    root_folder_path = sys.argv[1]

    # Check if the root folder path exists
    if not os.path.exists(root_folder_path):
        print(f"The root folder path '{root_folder_path}' does not exist.")
        sys.exit(1)

    # Create the output JSON path
    json_path = os.path.join(root_folder_path, "coco.json")

    # Convert the masks to COCO JSON
    convert_masks_to_coco_json(root_folder_path, json_path)

    print("Conversion completed. COCO JSON file saved at:", json_path)
