import os
from scipy.spatial.distance import euclidean
import pickle
import numpy as np
import torch
import logging
import time
from sam2.build_sam import build_sam2_video_predictor

# 设置日志记录
logging.basicConfig(filename='ped2_video_processing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def check_proximity(boxA, boxB, distance_threshold, iou_threshold=None):
    boxA = [int(coord) for coord in boxA]
    boxB = [int(coord) for coord in boxB]
    center_xA = (boxA[0] + boxA[2]) / 2
    center_yA = (boxA[1] + boxA[3]) / 2
    center_xB = (boxB[0] + boxB[2]) / 2
    center_yB = (boxB[1] + boxB[3]) / 2
    distance = euclidean((center_xA, center_yA), (center_xB, center_yB))
    if iou_threshold is None:
        return distance <= distance_threshold
    iou = calculate_iou(boxA, boxB)
    return distance <= distance_threshold or iou >= iou_threshold

device = torch.device("cuda:5")
sam2_checkpoint = "../checkpoints/sam2_hiera_base_plus.pt"
model_cfg = "sam2_hiera_b+.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

video_root_dir = "/data/cxli/yuzhi/Accurate-Interpretable-VAD/data/ped2/testing/frames"
box_data_root_dir = "/data/cxli/yuzhi/Accurate-Interpretable-VAD/anomalous_boxes"

all_video_segments = {}
total_frames = 0
total_time = 0

try:
    for video_dir_name in os.listdir(video_root_dir):
        video_dir_path = os.path.join(video_root_dir, video_dir_name)
        if not os.path.isdir(video_dir_path):
            continue

        logging.info(f"Processing video: {video_dir_name}")
        frame_names = sorted([
            p for p in os.listdir(video_dir_path)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ], key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_dir_path)
        predictor.reset_state(inference_state)

        box_data_path = os.path.join(box_data_root_dir, f"video_segment_{int(video_dir_name)}/video_segment_{int(video_dir_name)}_data.pkl")
        with open(box_data_path, 'rb') as file:
            box_data = pickle.load(file)

        iou_threshold = 0.3
        distance_threshold = 20
        center_distance_threshold = 20
        global_obj_id = 0
        frame_interval = 5

        frame_boxes = {}
        box_id_tracker = {}
        unassigned_obj_ids = []

        first_frame_idx = next((idx for idx, boxes in box_data.items() if boxes), None)
        if first_frame_idx is not None:
            first_frame_boxes = box_data[first_frame_idx]
            frame_boxes[first_frame_idx] = first_frame_boxes

            for boxA, score in first_frame_boxes:
                center_x = int((boxA[0] + boxA[2]) / 2)
                center_y = int((boxA[1] + boxA[3]) / 2)
                is_new_target = False

                for next_frame in range(first_frame_idx + 1, first_frame_idx + 6):
                    if next_frame in box_data:
                        for next_box, _ in box_data[next_frame]:
                            if check_proximity(boxA, next_box, distance_threshold, iou_threshold):
                                is_new_target = True
                                break
                        if is_new_target:
                            break

                boxA_key = tuple(boxA)
                if is_new_target:
                    box_id_tracker[boxA_key] = global_obj_id
                    points = np.array([(center_x, center_y)], dtype=np.float32)
                    labels = np.ones(1, dtype=np.int32)

                    start_time = time.time()
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=first_frame_idx,
                        obj_id=global_obj_id,
                        points=points,
                        labels=labels,
                        box=boxA,
                    )
                    inference_time = time.time() - start_time
                    total_time += inference_time
                    total_frames += 1
                    
                    logging.info(f"Initial frame: Added box, obj_id={global_obj_id} at frame {first_frame_idx}, center=({center_x}, {center_y}), score={score}, inference time={inference_time:.4f}s")
                    global_obj_id += 1

        for frame_idx, box_list in box_data.items():
            if frame_idx == first_frame_idx:
                continue

            frame_boxes[frame_idx] = box_list
            used_obj_ids = set()

            for boxA, score in box_list:
                center_x = int((boxA[0] + boxA[2]) / 2)
                center_y = int((boxA[1] + boxA[3]) / 2)
                boxA_key = tuple(boxA)
                inherited_obj_id = None
                for prev_frame in range(max(0, frame_idx - 5), frame_idx):
                    if prev_frame in box_data:
                        for prev_box, _ in box_data[prev_frame]:
                            prev_box_key = tuple(prev_box)
                            potential_obj_id = box_id_tracker.get(prev_box_key)
                            if check_proximity(boxA, prev_box, distance_threshold, iou_threshold) and potential_obj_id not in used_obj_ids:
                                inherited_obj_id = potential_obj_id
                                break
                        if inherited_obj_id is not None:
                            break

                if inherited_obj_id is not None:
                    box_id_tracker[boxA_key] = inherited_obj_id
                    used_obj_ids.add(inherited_obj_id)
                    logging.info(f"Frame {frame_idx}: Box {boxA} inherited obj_id={inherited_obj_id}")

            new_unassigned_obj_ids = []
            for remaining_obj_id, prev_box in unassigned_obj_ids:
                for boxA, score in box_list:
                    boxA_key = tuple(boxA)
                    if box_id_tracker.get(boxA_key) is None:
                        if check_proximity(boxA, prev_box, center_distance_threshold):
                            box_id_tracker[boxA_key] = remaining_obj_id
                            used_obj_ids.add(remaining_obj_id)
                            logging.info(f"Frame {frame_idx}: Box {boxA} inherited remaining obj_id={remaining_obj_id} based on center distance")
                            break
                else:
                    new_unassigned_obj_ids.append((remaining_obj_id, prev_box))

            unassigned_obj_ids = new_unassigned_obj_ids

            for boxA, score in box_list:
                boxA_key = tuple(boxA)
                if box_id_tracker.get(boxA_key) is None:
                    next_count = 0
                    for next_frame in range(frame_idx + 1, min(frame_idx + 6, max(box_data.keys()) + 1)):
                        if next_frame in box_data:
                            for next_box, _ in box_data[next_frame]:
                                if check_proximity(boxA, next_box, distance_threshold, iou_threshold):
                                    next_count += 1
                                    break

                    if next_count >= 3:
                        box_id_tracker[boxA_key] = global_obj_id
                        points = np.array([(center_x, center_y)], dtype=np.float32)
                        labels = np.ones(1, dtype=np.int32)

                        start_time = time.time()
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=global_obj_id,
                            points=points,
                            labels=labels,
                            box=boxA,
                        )
                        inference_time = time.time() - start_time
                        total_time += inference_time
                        total_frames += 1
                        
                        logging.info(f"Added new box, obj_id={global_obj_id} at frame {frame_idx}, center=({center_x}, {center_y}), score={score}, inference time={inference_time:.4f}s")
                        global_obj_id += 1

            if frame_idx % frame_interval == 0:
                for boxA, score in box_list:
                    center_x = int((boxA[0] + boxA[2]) / 2)
                    center_y = int((boxA[1] + boxA[3]) / 2)
                    points = np.array([(center_x, center_y)], dtype=np.float32)
                    labels = np.ones(1, dtype=np.int32)
                    boxA_key = tuple(boxA)
                    obj_id = box_id_tracker.get(boxA_key)

                    if obj_id is not None:
                        start_time = time.time()
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            points=points,
                            labels=labels,
                            box=boxA,
                        )
                        inference_time = time.time() - start_time
                        total_time += inference_time
                        total_frames += 1
                        
                        logging.info(f"Global hint: obj_id={obj_id} at frame {frame_idx}, center=({center_x}, {center_y}), score={score}, inference time={inference_time:.4f}s")

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: out_mask_logits[i].cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        all_video_segments[video_dir_name] = video_segments

except Exception as e:
    logging.error(f"An error occurred: {e}")
    # 保存已经处理好的视频分割结果
    with open('partial_video_segments.pkl', 'wb') as f:
        pickle.dump(all_video_segments, f)
    print("An error occurred. Partial results have been saved.")

# 计算并打印平均推理时间
if total_frames > 0:
    avg_inference_time = total_time / total_frames
    logging.info(f"Average inference time per frame: {avg_inference_time:.4f}s")
    print(f"Average inference time per frame: {avg_inference_time:.4f}s")

# 保存所有视频的分割结果
output_pickle_path = "all_video_segments_10.pkl"
with open(output_pickle_path, 'wb') as f:
    pickle.dump(all_video_segments, f)

print("All videos processed. Results are stored in 'all_video_segments' dictionary.")