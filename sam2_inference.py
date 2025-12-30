import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor
import pickle
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 保存字典
def save_results(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)

# 设置设备
device = torch.device("cuda:8")

# 初始化模型
sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# 定义视频目录和异常框目录
video_dir_base = "/data/cxli/yuzhi/Accurate-Interpretable-VAD/data/ped2/testing/frames"
pkl_dir = "/data/cxli/yuzhi/Accurate-Interpretable-VAD/anomalous_boxes"

# 获取所有视频目录
video_dirs = [d for d in os.listdir(video_dir_base) if os.path.isdir(os.path.join(video_dir_base, d))]
video_dirs.sort()
# 存储所有视频的分割结果
all_videos_segments = {}

for video_index, video_dir in enumerate(tqdm(video_dirs, desc='Processing videos')):
    video_path = os.path.join(video_dir_base, video_dir)
    print(video_path)
    file_path = os.path.join(pkl_dir, f"video_segment_{video_index+1}")
    
    # 初始化推断状态
    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)
    
    # 加载异常框数据
    with open(file_path, 'rb') as file:
        box_data = pickle.load(file)
    
    # 处理每一帧：不区分视频中的不同个体目标，而是把所有检测到的“异常”视为同一个对象（赋予一样的ID）
    video_segments = {}  # 存储当前视频的分割结果
    for cur_frame_index, boxes in box_data.items():
        if boxes:  # 检查该帧是否有非空的边界框
            cur_points = []
            for box in boxes:
                # 提取坐标并计算中心点
                x1, y1, x2, y2 = box[0]
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cur_points.append([center_x, center_y])
                cur_box = np.array([box[0]], dtype=np.float32)

                # 为该帧添加box提示
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=cur_frame_index,
                    obj_id=0,
                    box=cur_box
                )

            # 为该帧添加Point提示（中心点）
            cur_labels = np.ones(len(cur_points), dtype=np.int32)
            cur_points = np.array(cur_points, dtype=np.float32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=cur_frame_index,
                obj_id=0,
                points=cur_points,
                labels=cur_labels,
            )
        
    # 运行传播并收集当前视频的结果
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # 保存当前视频的结果
    all_videos_segments[video_index+1] = video_segments

# 打印结果
for video_dir, segments in all_videos_segments.items():
    print(f"Video {video_dir}:")
    for frame_idx, segment in segments.items():
        print(f"  Frame {frame_idx}:")
        for obj_id, mask in segment.items():
            print(f"    Object {obj_id}: {mask.shape}")

save_results(all_videos_segments, 'video_segments_results.pkl')