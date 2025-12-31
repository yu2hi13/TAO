Official PyTorch Implementation of  
[Track Any Anomalous Object: A Granular Video Anomaly Detection](https://arxiv.org/abs/2506.05175)

---

**Track Any Anomalous Object (TAO)** introduces a **Granular Video Anomaly Detection framework** that, for the first time, unifies the detection and localization of **multiple fine-grained anomalous objects** within a single end-to-end pipeline.

Unlike conventional video anomaly detection methods that assign anomaly scores densely to every pixel at each time step, TAO reformulates anomaly detection as a **pixel-level tracking problem of anomalous objects**. By explicitly linking anomaly scores to downstream tasks such as **image segmentation** and **video object tracking**, our framework eliminates the need for heuristic threshold selection and enables more accurate and robust anomaly localization, even in long and challenging video sequences.

---

# Getting Started

## 1 Data Preparation

### UCSDped2

Only the **Ped2** subset is required for the default experimental setup.

Dataset reference:  
[UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)

Download the dataset:

cd datasets
wget http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz

yaml
复制代码

---

## 2 Pre-processing

### 2.1 Extract Frames

Convert video clips into frame sequences:

cd data
python extract_frames.py

php
复制代码

Directory structure:

path_videos = "./data/{dataset}/{training/testing}_videos/"
path_frames = "./data/{dataset}/{training/testing}/frames/"

yaml
复制代码

---

### 2.2 Optical Flow Extraction

Optical flow is extracted using **FlowNet2.0**.

Reference:  
https://github.com/NVIDIA/flownet2-pytorch

#### (1) Install FlowNet2.0

cd pre_processing
bash install_flownet2.sh
cd ..

bash
复制代码

#### (2) Download Pre-trained FlowNet2 Weights

Download from:  
https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view

Place the weights in:

pre_processing/checkpoints/

css
复制代码

#### (3) Extract Optical Flow from Ped2 Frames

The extracted flows are saved to:

./data/ped2/{training/testing}/flows/

diff
复制代码

- Test frames:
python flow.py --dataset_name=ped2

diff
复制代码

- Training frames:
python flow.py --dataset_name=ped2 --train

yaml
复制代码

---

### 2.3 Object Detection

Bounding box annotations are stored as:

{dataset}_bboxes_train.npy
{dataset}_bboxes_test.npy

yaml
复制代码

#### (1) Directly Use Provided Detections (Recommended)

Precomputed object detection results are provided here:  
https://drive.google.com/drive/folders/1BnjzuwxyXio2sNU_4w7rlTw4PcURlq_R

Please place the files as follows:

- Ped2 → `./data/ped2`
- Avenue → `./data/avenue`
- ShanghaiTech → `./data/shanghaitech`

---

#### (2) Generate Bounding Boxes Yourself

> **Note**  
> The results are identical to those provided in option (1).

Install Detectron2:

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

yaml
复制代码

Download the ResNet50-FPN pre-trained weights:

wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
-P pre_processing/checkpoints/

csharp
复制代码

Run object detection:

python pre_processing/bboxes.py [--dataset_name] [--train]

makefile
复制代码

Example: extract training bounding boxes for Ped2:

python pre_processing/bboxes.py --dataset_name=ped2 --train

csharp
复制代码

Output file:

./data/ped2/ped2_bboxes_train.npy

bash
复制代码

Extract test bounding boxes:

python pre_processing/bboxes.py --dataset_name=ped2

csharp
复制代码

Output file:

./data/ped2/ped2_bboxes_test.npy

yaml
复制代码

---

### 2.4 Count Frames

Compute the number of frames per video clip and the cumulative frame index:

cd data
python count_frames.py

yaml
复制代码

---

## 3 Extract Anomalous Boxes

### 3.1 Feature Extraction

We extract:
- Motion velocity features
- Deep appearance features

Pose features (`pose.npy`) are provided.

python feature_extraction.py --dataset_name={dataset}

yaml
复制代码

---

### 3.2 Score Calibration

Compute calibration parameters for each feature representation:

python score_calibration.py --dataset_name={dataset}

yaml
复制代码

---

### 3.3 Evaluation

Run anomaly evaluation:

python evaluate.py --dataset_name={dataset} --sigma={sigma}

yaml
复制代码

Recommended values:
- Ped2 / Avenue: `sigma = 3`
- ShanghaiTech: `sigma = 7`

---

### 3.4 Extract Anomalous Bounding Boxes

For **Ped2**:

python getbox_ped2.py

yaml
复制代码

---

## 4 Robust Filtering and SAM2 Inference

### 4.1 Install SAM2

Please follow the official repository:  
https://github.com/facebookresearch/sam2

---

### 4.2 Robust Filtering

compute_{dataset}.py

yaml
复制代码

---

### 4.3 SAM2 Inference

sam2_inference.py

yaml
复制代码

Paths configuration:

frame data
video_dirs = ./data/{dataset}/testing/frames

anomalous regions
pkl_dir = ./outputs/getbox_results/

yaml
复制代码

---

## 5 Experiments

### Pixel-level Evaluation

cd Benchmark_Metrics/pixel_level

yaml
复制代码

Process results after robust filtering:

1. Compute metrics:

python vsresult_process.py

markdown
复制代码

Metrics include:
- AUROC
- AP
- AUPRO
- F1-score

2. Evaluate SAM2 outputs:

python sam2_evaluate.py

yaml
复制代码

---

### Object-level Evaluation

1. Extract detected anomaly trajectories:

segment-anything-2/get_box_results.py

markdown
复制代码

2. Extract ground-truth tracks:

segment-anything-2/get_gt.ipynb

markdown
复制代码

3. Compute **TBDC** and **RBDC**:

python compute_tbdc_rbdc.py
--tracks-path=PATH_TO_GT_TRACKS
--anomalies-path=PATH_TO_DETECTIONS
--num-frames=NUM_FRAMES

lua
复制代码

Track format:

track_id, frame_id, x_min, y_min, x_max, y_max

lua
复制代码

Detection format:

frame_id, x_min, y_min, x_max, y_max, anomaly_score

yaml
复制代码

---

## Installation

We recommend using **two separate environments**.

### Environment 1 (Before Robust Filtering and SAM2 Inference)

python==3.7
torch==1.12.0+cu102

yaml
复制代码

### Environment 2 (SAM2)

Follow the official SAM2 installation instructions.

---

## Citation

If you find this work useful, please cite:

@article{tao2025track,
title = {Track Any Anomalous Object: A Granular Video Anomaly Detection Framework},
author = {Author Names},
journal = {arXiv preprint arXiv:2506.05175},
year = {2025}
}

复制代码
