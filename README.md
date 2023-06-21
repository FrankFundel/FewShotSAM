# Few-Shot SAM

Fine-Tuning SAM in a robust and efficient way to disambiguate mask predictions using few examples.

# Steps

- Load and observe SA-1B dataset
- Build test set using multiple other datasets (embedded)
- Evaluate SAM and naive approach using CLIP and DINOv2

- Build Vector Database (image, masks, cutout_id) using best encoder of 3.
- Test sampling and retrieval process

- Build model (Gated Self-Attention)
- Define training procedure
- Train and evaluate model

- Experiment with different architectures (e.g. ControlNet)
- Experiment with retrieving before training
- Try out "Visual Prompt Engineering"

# Done

1. Downloaded six different datasets
2. Unified them by converting bitmasks to COCO json and split them for each image
3. Create custom dataset class for sampling image, point and corresponding mask
4. Create hf5 test set using SAM embeddings

# Commands

Download .tar files
`cat sa1b.txt | parallel -j 5  --colsep $'\t'  wget -nc  -c  {2}    -O {1}`

Untar into folder
`find . -name '*.tar' -exec tar -xf {} -C sa1b \;`

Download from google drive:
`gdown https://drive.google.com/uc?id=1GrsDXtV0Osn0aZw0c-Wgb6reiaXp7OTl`

Fast quiet unzip:
`jar xf FSSAM\ Datasets.zip`

Start interactive job
`salloc -p gpu_4_a100 -n 1 -t 1:00:00 --mem=16000 --gres=gpu:1`

# Datasets

- EgoHos: images, masks
- GTEA: images, masks
- ZeroWaste-f: images, masks
- LVIS: images, json
- NDIS Park: images, json
- OVIS: images, json
- TrashCan: images, json