# Official PyTorch Implementation: Track Any Anomalous Object (TAO)

**[Track Any Anomalous Object: A Granular Video Anomaly Detection Framework](https://arxiv.org/abs/2506.05175)** **(CVPR 2025)**

---


## 📖 Introduction

**Track Any Anomalous Object (TAO)** introduces a **Granular Video Anomaly Detection framework** that, for the first time, unifies the detection and localization of **multiple fine-grained anomalous objects** within a single end-to-end pipeline.

Unlike conventional video anomaly detection methods that assign anomaly scores densely to every pixel at each time step, TAO reformulates anomaly detection as a **pixel-level tracking problem of anomalous objects**. By explicitly linking anomaly scores to downstream tasks such as **image segmentation** and **video object tracking**, our framework eliminates the need for heuristic threshold selection. This enables more accurate and robust anomaly localization, even in long and challenging video sequences.

---

## 🚀 Getting Started

### 1. Data Preparation

#### UCSDped2

Only the **Ped2** subset is required for the default experimental setup.

* **Dataset Reference:** [UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)

**Download the dataset:**

```bash
cd datasets
wget http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
```

---

### 2. Pre-processing

#### 2.1 Extract Frames

Convert video clips into frame sequences.

```bash
cd data
python extract_frames.py

```

**Directory Structure:**

```text
path_videos = "./data/{dataset}/{training/testing}_videos/"
path_frames = "./data/{dataset}/{training/testing}/frames/"
```

#### 2.2 Optical Flow Extraction

Optical flow is extracted using **FlowNet2.0**. (Reference: [FlowNet2 PyTorch](https://github.com/NVIDIA/flownet2-pytorch))

**(1) Install FlowNet2.0**

```bash
cd pre_processing
bash install_flownet2.sh
cd ..
```

**(2) Download Pre-trained Weights**
Download `FlowNet2_checkpoint.pth.tar` from [here](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view) and place it in:

```text
pre_processing/checkpoints/

```

**(3) Extract Optical Flow**
The extracted flows will be saved to `./data/ped2/{training/testing}/flows/`.

* **Test frames:**
```bash
python flow.py --dataset_name=ped2

```


* **Training frames:**
```bash
python flow.py --dataset_name=ped2 --train

```



#### 2.3 Object Detection

Bounding box annotations are stored as `{dataset}_bboxes_train.npy` and `{dataset}_bboxes_test.npy`.

**Option 1: Use Provided Detections (Recommended)**
Download precomputed results from [Google Drive](https://drive.google.com/drive/folders/1BnjzuwxyXio2sNU_4w7rlTw4PcURlq_R). Place files as follows:

* Ped2 → `./data/ped2`
* Avenue → `./data/avenue`
* ShanghaiTech → `./data/shanghaitech`

**Option 2: Generate Bounding Boxes Yourself**

> **Note:** The results are identical to Option 1.

1. **Install Detectron2:**
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

```


2. **Download ResNet50-FPN Weights:**
```bash
wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl -P pre_processing/checkpoints/

```


3. **Run Extraction:**
* **Training Set (e.g., Ped2):**
```bash
python pre_processing/bboxes.py --dataset_name=ped2 --train
# Output: ./data/ped2/ped2_bboxes_train.npy

```


* **Test Set:**
```bash
python pre_processing/bboxes.py --dataset_name=ped2
# Output: ./data/ped2/ped2_bboxes_test.npy

```





#### 2.4 Count Frames

Compute the number of frames per video clip and the cumulative frame index.

```bash
cd data
python count_frames.py

```

---

### 3. Extract Anomalous Boxes

#### 3.1 Feature Extraction

Extract motion velocity and deep appearance features. Pose features (`pose.npy`) are already provided.

```bash
python feature_extraction.py --dataset_name={dataset}

```

#### 3.2 Score Calibration

Compute calibration parameters for each feature representation.

```bash
python score_calibration.py --dataset_name={dataset}

```

#### 3.3 Evaluation

Run anomaly evaluation. Recommended sigma values:

* **Ped2 / Avenue:** `sigma = 3`
* **ShanghaiTech:** `sigma = 7`

```bash
python evaluate.py --dataset_name={dataset} --sigma={sigma}

```

#### 3.4 Extract Anomalous Bounding Boxes (Ped2)

```bash
python getbox_ped2.py

```

---

### 4. Robust Filtering and SAM2 Inference

#### 4.1 Install SAM2

Please follow the instructions in the [official SAM2 repository](https://github.com/facebookresearch/sam2).

#### 4.2 Robust Filtering

```bash
python compute_{dataset}.py

```

#### 4.3 SAM2 Inference

```bash
python sam2_inference.py

```

**Configuration Paths:**

* Frame data: `video_dirs = ./data/{dataset}/testing/frames`
* Anomalous regions: `pkl_dir = ./outputs/getbox_results/`

---

## 🧪 Experiments

### Pixel-level Evaluation

Navigate to the benchmark folder:

```bash
cd Benchmark_Metrics/pixel_level

```

**1. Process results after robust filtering:**
Computes AUROC, AP, AUPRO, and F1-score.

```bash
python vsresult_process.py

```

**2. Evaluate SAM2 outputs:**

```bash
python sam2_evaluate.py

```

### Object-level Evaluation

1. **Extract detected anomaly trajectories:**
```bash
python segment-anything-2/get_box_results.py

```


2. **Extract ground-truth tracks:**
Run the notebook: `segment-anything-2/get_gt.ipynb`
3. **Compute TBDC and RBDC:**
```bash
python compute_tbdc_rbdc.py \
  --tracks-path=PATH_TO_GT_TRACKS \
  --anomalies-path=PATH_TO_DETECTIONS \
  --num-frames=NUM_FRAMES

```



**Data Formats:**

* **Track:** `track_id, frame_id, x_min, y_min, x_max, y_max`
* **Detection:** `frame_id, x_min, y_min, x_max, y_max, anomaly_score`

---

## ⚙️ Installation

We recommend using **two separate environments** to avoid dependency conflicts.

### Environment 1 (Preprocessing & Initial Pipeline)

Used for sections 1 through 3.

* **Python:** 3.7
* **PyTorch:** 1.12.0+cu102

### Environment 2 (SAM2)

Used for Section 4 onwards.

* Follow the [official SAM2 installation instructions](https://github.com/facebookresearch/sam2).

---

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{tao2025track,
  title   = {Track Any Anomalous Object: A Granular Video Anomaly Detection Framework},
  author  = {Author Names},
  journal = {arXiv preprint arXiv:2506.05175},
  year    = {2025}
}

```
