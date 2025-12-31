# Official PyTorch Implementation: Track Any Anomalous Object (TAO)

**[Track Any Anomalous Object: A Granular Video Anomaly Detection Framework](https://arxiv.org/abs/2506.05175)**

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
wget [http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz](http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz)

#### 2.2 Optical Flow Extraction
Optical flow is extracted using **FlowNet2.0**. (Reference: [FlowNet2 PyTorch](https://github.com/NVIDIA/flownet2-pytorch))

**(1) Install FlowNet2.0**
```bash
cd pre_processing
bash install_flownet2.sh
cd ..
