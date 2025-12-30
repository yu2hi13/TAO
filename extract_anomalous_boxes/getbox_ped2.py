import numpy as np
import argparse
import faiss
from video_dataset import VideoDatasetWithFlows, img_tensor2numpy
import os
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import sys
import cv2
import torch
import pickle



def draw_bounding_box(image, box, color, thickness=2):
    """在图像上绘制边框"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def is_box_in_list(box, box_list):
    """检查边界框是否在列表中"""
    for b in box_list:
        if all(np.equal(box, b)):
            return True
    return False

def boxes_overlap(box1, box2):
    """检查两个边界框是否有重叠"""
    x1, y1, x2, y2 = box1
    x1_other, y1_other, x2_other, y2_other = box2

    return not (x2 < x1_other or x1 > x2_other or y2 < y1_other or y1 > y2_other)

def merge_boxes(boxes):
    """合并重叠的边界框"""
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)

def evaluate(args, root):
    test_clip_lengths = np.load(os.path.join(root, args.dataset_name, 'test_clip_lengths.npy'))
    print(test_clip_lengths)
    
    bbox_data = np.load('/data/cxli/yuzhi/Accurate-Interpretable-VAD/data/ped2/ped2_bboxes_test.npy', allow_pickle=True)

    train_velocity = np.load('extracted_features/{}/train/velocity.npy'.format(args.dataset_name), allow_pickle=True)
    train_velocity = np.concatenate(train_velocity, 0)
    train_deep_features = np.load('extracted_features/{}/train/deep_features.npy'.format(args.dataset_name), allow_pickle=True)
    train_deep_features = np.concatenate(train_deep_features, 0)

    train_pose = np.load('extracted_features/{}/train/pose.npy'.format(args.dataset_name), allow_pickle=True)
    without_empty_frames = []
    for i in tqdm(range(len(train_pose))):
        if len(train_pose[i]):
            without_empty_frames.append(train_pose[i])
    train_pose = np.concatenate(without_empty_frames, 0)

    test_velocity = np.load('extracted_features/{}/test/velocity.npy'.format(args.dataset_name), allow_pickle=True)
    test_pose = np.load('extracted_features/{}/test/pose.npy'.format(args.dataset_name), allow_pickle=True)
    test_deep_features = np.load('extracted_features/{}/test/deep_features.npy'.format(args.dataset_name), allow_pickle=True)

    test_dataset = VideoDatasetWithFlows(dataset_name=args.dataset_name, root=root,
                                         train=False, sequence_length=0, all_bboxes=None, normalize=False, mode='last')

    if args.dataset_name == 'ped2':
        velocity_density_estimator = GaussianMixture(n_components=2, random_state=0).fit(train_velocity)
    else:
        velocity_density_estimator = GaussianMixture(n_components=5, random_state=0).fit(train_velocity)

    # 使用 GPU 版本的 Faiss 索引
    res = faiss.StandardGpuResources()
    gpu_index = 6 # 使用 GPU 0

    index = faiss.IndexFlatL2(train_deep_features.shape[1])
    index_deep_features = faiss.index_cpu_to_gpu(res, gpu_index, index)
    index_deep_features.add(train_deep_features.astype(np.float32))

    index = faiss.IndexFlatL2(train_pose.shape[1])
    index_pose = faiss.index_cpu_to_gpu(res, gpu_index, index)
    index_pose.add(train_pose.astype(np.float32))
    # 计算训练数据的最小值和最大值
    min_velocity_train = np.min(train_velocity)
    max_velocity_train = np.max(train_velocity)
    min_deep_features_train = np.min(train_deep_features)
    max_deep_features_train = np.max(train_deep_features)

    test_pose_scores = []

    # 输出保存路径
    output_folder = 'anomalous_boxes_ped2_5'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_segment in range(len(test_clip_lengths)):
        anomalous_data = {}
        # 为每个视频段创建一个单独的子文件夹
        segment_folder = os.path.join(output_folder, f'video_segment_{video_segment+1}')
        if not os.path.exists(segment_folder):
            os.makedirs(segment_folder)

        if video_segment == 0:
            start_frame = 0
        else:
            start_frame = test_clip_lengths[video_segment - 1]
        end_frame = test_clip_lengths[video_segment] if video_segment < len(test_clip_lengths) - 1 else len(test_dataset)

        for i in tqdm(range(start_frame, end_frame), total=end_frame - start_frame):
            cur_pose = test_pose[i]
            cur_velocity = test_velocity[i]
            cur_deep_features = test_deep_features[i]

            object_scores = {"velocity": [], "pose": [], "deep_features": []}

            if cur_pose.shape[0]:
                D, I = index_pose.search(cur_pose.astype(np.float32), 1)
                score_pose = np.mean(D, axis=1)
                max_score_pose = np.max(score_pose)
                test_pose_scores.append(max_score_pose)
                object_scores['pose'] = score_pose
            else:
                test_pose_scores.append(0)
                object_scores['pose'] = [0]

            D, I = index_deep_features.search(cur_deep_features.astype(np.float32), 1)
            score_features = np.mean(D, axis=1)
            object_scores['deep_features'] = score_features
            object_scores['velocity'] = -velocity_density_estimator.score_samples(cur_velocity)

            if test_dataset.all_gt[i] == 1:
                if len(object_scores['velocity']) > 0 and len(object_scores['deep_features']) > 0:
                    # # 使用测试数据本身进行标准化
                    # min_velocity = np.min(object_scores['velocity'])
                    # max_velocity = np.max(object_scores['velocity'])
                    # min_deep_features = np.min(object_scores['deep_features'])
                    # max_deep_features = np.max(object_scores['deep_features'])

                    # # 使用 Min-Max Scaling 进行标准化
                    # norm_velocity = (object_scores['velocity'] - min_velocity) / (max_velocity - min_velocity)
                    #norm_deep_features = (object_scores['deep_features'] - min_deep_features_train) / (max_deep_features_train - min_deep_features_train)
                    epsilon = 1e-8
                    norm_velocity = (object_scores['velocity'] - min_velocity_train) / (max_velocity_train - min_velocity_train + epsilon)
                    norm_deep_features = (object_scores['deep_features'] - min_deep_features_train) / (max_deep_features_train - min_deep_features_train + epsilon)


                    # 计算总分数
                    total_scores = norm_velocity + norm_deep_features

                    threshold =9 
                    high_score_indices = np.where(total_scores > threshold)[0]
            
                    high_score_boxes = [bbox_data[i][idx] for idx in high_score_indices]
                    high_score_values = [total_scores[idx] for idx in high_score_indices]  # 获取对应的分数

                    merged_boxes = []
                    merged_scores = []

                    img = test_dataset[i][0].squeeze(0)
                    img = img_tensor2numpy(img)

                    original_size = (360, 240)
                    img_resized = cv2.resize(img, original_size)
                    img_with_boxes = img_resized.copy()

                    for box in high_score_boxes:
                        img_with_boxes = draw_bounding_box(img_with_boxes, box, color=(0, 255, 0))  # 使用绿色框

                    # 计算该帧在视频段中的相对帧序号
                    relative_frame_index = i - start_frame

                    # 保存带有检测框的图像，并使用相对帧序号命名文件
                    img_save_path = os.path.join(segment_folder, f"anomalous_frame_{relative_frame_index+1}.png")
                    cv2.imwrite(img_save_path, img_with_boxes)

                    for current_box, score in zip(high_score_boxes, high_score_values):
                        box_added = False
                        boxes_to_merge = [current_box]
                        
                        for j, merged_box in enumerate(merged_boxes):
                            if boxes_overlap(current_box, merged_box):
                                boxes_to_merge.append(merged_box)
                                merged_boxes.pop(j)
                                merged_scores.pop(j)
                                break
                        else:
                            merged_boxes.append(current_box)
                            merged_scores.append(score)
                            box_added = True

                        if not box_added and len(boxes_to_merge) > 1:
                            merged_box = merge_boxes(boxes_to_merge)
                            merged_boxes.append(merged_box)
                            merged_scores.append(np.mean([score for _, score in zip(boxes_to_merge, high_score_values)]))

                    if relative_frame_index not in anomalous_data:
                        anomalous_data[relative_frame_index] = []
                    anomalous_data[relative_frame_index].extend(list(zip(merged_boxes, merged_scores)))

        if len(anomalous_data) > 0:
            with open(os.path.join(segment_folder, f'video_segment_{video_segment+1}_data.pkl'), 'wb') as f:
                pickle.dump(anomalous_data, f)
        else:
            print(f"视频段 {video_segment} 没有异常数据。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    parser.add_argument("--sigma", type=int, default=3, help='sigma for gaussian1d smoothing')
    args = parser.parse_args()
    root = 'data/'
    evaluate(args, root)