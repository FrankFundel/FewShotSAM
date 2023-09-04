# Few-Shot SAM

Fine-Tuning SAM in a robust and efficient way to disambiguate mask predictions using few examples.

# ToDo

15. Build training dataset
16. Build vector database (image, masks, cutout_id)
17. Test sampling and retrieval process

18. Build model (Gated Self-Attention)
19. Define training procedure
20. Train and evaluate model

23. Try out Visual Prompt Engineering

# Done

1. Downloaded six different datasets
2. Unified them by converting bitmasks to COCO json and split them for each image
3. Embed each test image using SAM encoder
4. Create custom dataset class for sampling image, embedding point and corresponding mask
5. Baseline test + oracle

6. Download Liebherr Dataset
7. Convert to COCO
8. Embed, explore, baseline test
9. Fine-tune SAM on Liebherr Dataset and test + oracle
10. Compare annotation time

11. Functions to create example datasets from cutouts
12. Test dual-stage method on all datasets
13. Try out Visual Prompt Engineering and Mask Prompt Tuning
14. Try out DINOv2 and CLIP


# Commands

Download .tar files
`cat sa1b.txt | parallel -j 5 --colsep $'\t' wget -nc -c {2} -O {1}`

Untar into folder
`find . -name '*.tar' -exec tar -xf {} -C sa1b \;`

Download from google drive
`gdown https://drive.google.com/uc?id=1GrsDXtV0Osn0aZw0c-Wgb6reiaXp7OTl`

Fast quiet unzip
`jar xf FSSAM\ Datasets.zip`

Start interactive job
`salloc -p gpu_4_a100 -n 1 -t 1:00:00 --mem=16000 --gres=gpu:1`

List workspaces
`ws_list`

Masks to coco.json
`python masks_to_coco_jsons.py "/pfs/work7/workspace/scratch/ul_xto11-FSSAM/FSSAM Datasets/GTEA"`

coco.json to individual .jsons
`python coco_json_split.py "/pfs/work7/workspace/scratch/ul_xto11-FSSAM/FSSAM Datasets/GTEA"`

Embed images (uses custom transform functions and dataset because images are not of the same size)
`python prepare_dataset.py --dataset_path "/pfs/work7/workspace/scratch/ul_xto11-FSSAM/FSSAM Datasets/LVIS/dataset"`

Move all .jpeg files in one directory recursively, such that x/y/z/file.jpg becomes dataset/images/z_file.jpg
`find . -name '*.jpg' -exec bash -c '
    for filepath; do
        filename=$(basename "$filepath")
        dirname=$(dirname "$filepath")
        parentdir=$(basename "$dirname")
        new_filename="dataset/images/${parentdir}_${filename}"
        mv "$filepath" "$new_filename"
    done
' bash {} +`

Convert Label Studio to COCO
`python ls_to_coco_json.py /pfs/work7/workspace/scratch/ul_xto11-FSSAM/Liebherr/batch_01_small_batch_02_small.json /pfs/work7/workspace/scratch/ul_xto11-FSSAM/Liebherr/config.xml`

# Datasets

- EgoHos: images, masks
- GTEA: images, masks
- ZeroWaste-f: images, masks
- LVIS: images, json
- NDIS Park: images, json
- OVIS: images, json
- TrashCan: images, json
- Liebherr: images, json