import json
import os
import tqdm
import sys

def split_coco_json(input_file, output_dir):
    with open(input_file, 'r') as f:
        coco_data = json.load(f)
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Iterate over each image in the COCO JSON
        for image in tqdm.tqdm(coco_data['images']):
            image_id = image['id']
            if 'file_name' in image:
                if image['file_name'] is not None:
                    file_name = image['file_name'][:-3]
            else:
                file_name = str(image['id']).zfill(12) + '.'
            
            if file_name is None:
                continue
            
            annotations = []
            
            # Find the corresponding annotations for the current image
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == image_id:
                    annotations.append(annotation)
            
            # Create a new JSON object for the current image and annotations
            output_data = {
                'image': image,
                'annotations': annotations
            }
            
            # Write the JSON object to a new file
            output_file = os.path.join(output_dir, f'{file_name}json')
            with open(output_file, 'w') as out_f:
                json.dump(output_data, out_f, indent=4)

if __name__ == "__main__":
    # Check if the root folder path argument is provided
    if len(sys.argv) != 2:
        print("Please provide the root folder path as an argument.")
        print("Usage: python coco_json_split.py /path/to/root/folder")
        sys.exit(1)

    # Retrieve the root folder path from the command-line argument
    root_folder_path = sys.argv[1]

    # Check if the root folder path exists
    if not os.path.exists(root_folder_path):
        print(f"The root folder path '{root_folder_path}' does not exist.")
        sys.exit(1)
    
    # Usage example
    input_file = os.path.join(root_folder_path, 'coco.json')
    output_dir = os.path.join(root_folder_path, 'dataset/images')
    split_coco_json(input_file, output_dir)
    
    print('Splitting completed successfully.')
