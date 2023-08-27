import os
import sys
import json
import numpy as np
import base64
import tqdm
import cv2
from pycocotools import mask as mask_utils
import xml.etree.ElementTree as ET

def extract_labels_from_xml(xml_file_path):
    labels_dict = {}

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    polygon_labels = root.find(".//PolygonLabels")
    if polygon_labels is not None:
        for i, label_element in enumerate(polygon_labels.findall("Label")):
            label_value = label_element.get("value")
            labels_dict[label_value] = i

    return labels_dict

def convert_labelstudio_to_coco_json(labelstudio_json_path, config_path, coco_json_path):
    with open(labelstudio_json_path, "r") as json_file:
        labelstudio_data = json.load(json_file)
    
    config = extract_labels_from_xml(config_path)
    
    images_path = labelstudio_json_path.split('.')[0]

    # Create a dictionary for the COCO JSON format
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": list(config.keys())
    }

    ids = 0
    for i, image_info in tqdm.tqdm(enumerate(labelstudio_data)):
        file_upload = image_info["file_upload"]
        file_upload = file_upload[(file_upload.find('-')+1):]
        file_name = None
        for file in os.listdir(images_path):
            if file.endswith(file_upload):
                file_name = file
                break
        if file_name is None:
            print("File not found", file_upload)
            continue
        
        width = 0
        height = 0
        for annot in image_info["annotations"][0]["result"]:
            points = np.asarray(annot["value"]["points"])
            label = annot["value"]["polygonlabels"][0].replace('"', '')
            width = annot["original_width"]
            height = annot["original_height"]
            
            points[:, 0] *= width/100
            points[:, 1] *= height/100
            points = points.astype(int)
            
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(binary_mask, pts=[points], color=1)
            binary_mask = np.asfortranarray(binary_mask)

            # Encode the binary mask
            encoded_mask = mask_utils.encode(binary_mask)
            encoded_mask["counts"] = encoded_mask["counts"].decode("utf-8")
            
            # Create an annotation entry for the image
            annotation = {
                "id": ids,
                "image_id": i + 1,
                "category_id": config[label],
                "segmentation": [encoded_mask],
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation)
            ids += 1

        # Add the image and annotation entries to the COCO data
        coco_data["images"].append({
            "id": i + 1,
            "file_name": file_name,
            "height": height,
            "width": width,
        })

    # Save the COCO JSON file
    with open(coco_json_path, "w") as json_file:
        json.dump(coco_data, json_file)


if __name__ == "__main__":
    # Check if the path arguments are provided
    if len(sys.argv) != 3:
        print("Please provide the labelstudio json path as an argument.")
        print("Please provide the config.xml path as an argument.")
        print("Usage: python ls_to_coco.py /path/to/json /path/to/config.xml")
        sys.exit(1)

    # Retrieve the paths from the command-line argument
    labelstudio_json_path = sys.argv[1]
    config_path = sys.argv[2]

    # Check if the paths exists
    if not os.path.exists(labelstudio_json_path):
        print(f"The labelstudio json path '{labelstudio_json_path}' does not exist.")
        sys.exit(1)
    if not os.path.exists(config_path):
        print(f"The config path '{config_path}' does not exist.")
        sys.exit(1)

    # Create the output JSON path
    json_path = os.path.join(os.path.dirname(labelstudio_json_path), "coco.json")

    # Convert the masks to COCO JSON
    convert_labelstudio_to_coco_json(labelstudio_json_path, config_path, json_path)

    print("Conversion completed. COCO JSON file saved at:", json_path)
