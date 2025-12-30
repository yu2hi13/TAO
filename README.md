
Official PyTorch Implementation of [Track Any Anomalous Object: A Granular Video Anomaly Detection](https://arxiv.org/abs/2506.05175)


![[Pasted image 20251210200537.png]]

**Track Any Anomalous Object (TAO)** introduces a Granular Video Anomaly Detection Framework that, for the first time, integrates the detection of multiple fine-grained anomalous objects into a unified framework. Unlike methods that assign anomaly scores to every pixel at each moment, our approach transforms the problem into pixel-level tracking of anomalous objects. By linking anomaly scores to subsequent tasks such as image segmentation and video tracking, our method eliminates the need for threshold selection and achieves more precise anomaly localization, even in long and challenging video sequences.

# Getting Started

## 1 Data preparation

### UCSDped2
	just need ped2

[UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)


```
cd datasets

wget http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
```



## 2 Pre-processing

### 2.1 Extract frames 

clipp the video to frames

```
cd data
python extract_frames.py
```

```
path_videos = "./data/{datasets}}/{training/testing}_videos/"

path_frames =  "./data/{datasets}}/{training/testing}/frames"
```

### 2.2 Optical flows

extract optical flows in videos using use [FlowNet2.0](https://github.com/NVIDIA/flownet2-pytorch)


（1）install  [FlowNet2.0](https://github.com/NVIDIA/flownet2-pytorch). ：

```
cd pre_processing
bash install_flownet2.sh
cd ..
```

（2）download the pre-trained FlowNet2 weights :

from [here](//drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)  and place it in `pre_processing/checkpoints`

（3）extract flows from Ped2 frames：

save the results to `./data/ped2/training/flows/`

- test frames：
```
  python flow.py --dataset_name=ped2
  ```
- train frames：
```
  python flow.py --dataset_name=ped2 --train
  ```

### 2.3 Object detection

	bounding boxes {datasets}_bboxes_test/train.npy

#### （1）Directly use

Our object detector outputs are provided [here](https://drive.google.com/drive/folders/1BnjzuwxyXio2sNU_4w7rlTw4PcURlq_R?usp=sharing). Set up the bounding boxes by placing the corresponding files in the following folders:

- All files for Ped2 should be placed in:  `./data/ped2`
- All files for Avenue should be placed in:  `./data/avenue`
- All files for ShanghaiTech should be placed in:  `./data/shanghaitech`

---
#### （2）Prepare by yourself

> [!TIP]
> the results are same as (1)

This section describes how to prepare the object detector to extract bounding boxes:

Please install the Detectron2 library by executing the following commands:

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Then download the ResNet50-FPN weights by executing:

```
wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl -P pre_processing/checkpoints/
```

Run the following command to detect all the foreground objects.

```
python pre_processing/bboxes.py [--dataset_name] [Optional: --train] 
```

E.g., In order to extract all train objects from Ped2:

```
python pre_processing/bboxes.py --dataset_name=ped2 --train 
```

This will save the results to `./data/ped2/ped2_bboxes_train.npy`, where each item contains all the bounding boxes in a single video frame.

In order to extract all test objects from Ped2:

```
python pre_processing/bboxes.py --dataset_name=ped2
```

This will save the results to `./data/ped2/ped2_bboxes_test.npy`, where each item contains all the bounding boxes in a single video frame.

### 2.4 Count frames

Count the number of frames for each video clip and calculate the cumulative frame index

```
cd data
python count_frames.py
```

## 3 Extract anomalous boxes

### 3.1 Feature extraction

extract velocity features and deep features
we have provided the pose features (pose.npy)

```
python feature extraction --{datasets}
```

### 3.2 Score calibration

To compute calibration parameters for each representation, run the following command:

```
python score_calibration.py [--dataset_name]
```

### 3.3 Evaluation

you can evaluate by running the following command:

```
python evaluate.py [--dataset_name] [--sigma]
```

We usually use `--sigma=3` for Ped2 and Avenue, and `--sigma=7` for ShanghaiTech.

### 3.4 Get anomalous boxes

- **ped2：**
```
python getbox_ped2.py
``````

## 4 Robust filtering and SAM2 inference

### 4.1 Install SAM2

[SAM2](https://github.com/facebookresearch/sam2.git)

### 4.2 **Robust filtering**

`compute_{datasets}.py` 


### 4.3 SAM2 inference

`sam2_inference.py`

```
# frame dates
video_dirs = ./data/{datasets}/testing/frames

# anomalous dates
pkl_dir = ./{from getbox_ped2.py's output}
```

## 5 Experiment 

###  **Pixel-level**


`cd Benchmark_Metrics/pixel_level`

process the dates after robust filtering

（1）`python vsresult_process.py` 

to get  **AUROC AP AUPRO F1**

（2）`python sam2_evaluate.py` 


### **Object-level**


（1）`segment-anything-2/get_box_results.py` 
to get anomalies-path（video_segments_boxes_sam_10）

（2）`segment-anything-2/get_gt.ipyb` 
to get tracks-path

（3）compute **TBDC** and **RBDC**

```
python compute_tbdc_rbdc.py --tracks-path= --anomalies-path= --num-frames=
```

- `tracks-path` is the path to the folder containing the tracks for all videos.
    - The tracks are organized as follows:
        - for each video, we have a txt file containing all the regions with the following format:
            
            track_id, frame_id, x_min, y_min, x_max, y_max
            
        - the track_id and frame_id must be in ascending order
            
- `anomalies-path` is the path to the folder containing the detected anomaly regions for all videos.
    - The anomaly regions are organized as follows:
        - for each video, we have a txt file containing all the detected regions with the following format:
            
            frame_id, x_min, y_min, x_max, y_max, anomaly_score
            
- `num-frames` is the total number of frames in the videos.
- The name of the video tracks must match the name of the detected region per video.

# Installation

It's recommended to create a two discrete environments .

(1) Before [[#4 Robust filtering and SAM2 inference]] :

```
python==3.7
torch==1.12.0+cu102
```

(2)  Follow [SAM2](https://github.com/facebookresearch/sam2.git)

# Citation

If you find this useful, please cite our paper:
```
@article{




}
```
