import os
import re
import math
import json
import argparse
import warnings
import traceback

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class MVBenchDataset(Dataset):
    def __init__(self, data_list, processor, num_segments=8):
        self.data_list = data_list
        self.processor = processor
        self.num_segments = num_segments

    def __len__(self):
        return len(self.data_list)

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_video(self, video_path, bound=None):
        # return {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0}
        return {"type": "video", "video": video_path, "fps": 1.0}

    def read_gif(self, video_path, bound=None):
        # return {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0}
        return {"type": "video", "video": video_path, "fps": 1.0}

    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1)
        images_group = [os.path.join(video_path, f"{i:05d}.jpg") for i in frame_indices]
        # return [{"type": "image", "image": img, "min_pixels": 50176, "max_pixels": 50176} for img in images_group]
        return [{"type": "image", "image": img} for img in images_group]

    def __getitem__(self, idx):
        item = self.data_list[idx]
        data_type = item['data_type']
        bound = (item['data']['start'], item['data']['end']) if item['bound'] else None
        video_path = os.path.join(item['prefix'], item['data']['video'])

        if data_type in ['video', 'gif']:
            visual_input = self.read_video(video_path, bound)
        else:  # frame
            visual_input = self.read_frame(video_path, bound)

        question = item['data']['question']
        options = item['data']['candidates']
        answer = item['data']['answer']
        task_type = item['task_type']

        answer_idx = -1
        letters = []
        options_string = ''
        for option_idx, c in enumerate(options):
            letters.append(f"{chr(ord('A') + option_idx)}")
            options_string += f"({chr(ord('A') + option_idx)}) {c}\n"
            if c == answer:
                answer_idx = option_idx

        instruct = f'Question: {question}\nOptions:\n{options_string}Answer with the option\'s letter from the given choices directly and only give the best option.'

        return {
            'visual_input': visual_input,
            'video_path': video_path,
            'instruct': instruct,
            'letters': ','.join(letters),
            'answer_idx': answer_idx,
            'task_type': task_type
        }

tasks = {
    "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/", "frame", True),
    "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True),
    "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True),
    "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/", "video", True),
    "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "sta/sta_video/", "video", True),
    "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
    "Character Order": ("character_order.json", "perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False)
}

def build_mvbench_eval(args, processor, num_frames):
    data_list = []
    for task_name, task in tasks.items():
        json_file = os.path.join(args.question_file, task[0])
        vis_folder = os.path.join(args.video_folder, task[1])
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        for data in json_data:
            data_list.append({
                'task_type': task_name,
                'prefix': vis_folder,
                'data_type': task[2],
                'bound': task[3],
                'data': data
            })
    data_list = get_chunk(data_list, args.num_chunks, args.chunk_idx)
    dataset = MVBenchDataset(data_list, processor, num_segments=num_frames)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return dataloader

def mvbench_dump(ans_file, line, outputs):
    for idx, output in enumerate(outputs):
        vid = line['video_path'][idx]
        instruct = line['instruct'][idx]
        task_type = line['task_type'][idx]
        letters = line['letters'][idx].split(',')
        answer_idx = line['answer_idx'][idx].item()

        pred_answer = re.findall(f'[\(,\ ]*[{letters[0]}-{letters[-1]}][\),\ ]*', output)
        try:
            assert len(pred_answer) >= 1, 'The video \"{}\" output \"{}\" is not in the expected format'.format(line['video_path'], instruct + '\n' + output)
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
        except:
            traceback.print_exc()
            pred_idx = 2

        ans_file.write(json.dumps({"vid": vid, "task_type": task_type, "pred": pred_idx, "gt": answer_idx}) + '\n')
def run_inference(args):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    val_loader = build_mvbench_eval(args, processor, args.num_frames)

    for i, line in enumerate(tqdm(val_loader)):
        visual_input = line['visual_input']
        instruct = line['instruct'][0]

        # Prepare content based on input type
        if 'type' in visual_input and visual_input['type'][0] == 'video':
            # Video input
            content = [
                {
                    "type": "video",
                    "video": visual_input['video'][0],
                    # "max_pixels": 360 * 420,
                    "fps": float(visual_input['fps'][0])
                }
            ]
        elif isinstance(visual_input, list):
            # Image input (multiple frames)
            content = [
                {
                    "type": "image",
                    "image": img['image'][0]
                    # "min_pixels": 50176,
                    # "max_pixels": 50176
                } for img in visual_input
            ]
        else:
            raise ValueError(f"Unexpected visual input format: {visual_input}")

        # Add text instruction
        content.append({"type": "text", "text": instruct})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        print(messages)
        # Process input for Qwen2-VL
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        mvbench_dump(ans_file, line, [output])

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='Path to the Qwen2-VL model', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to extract from each video")
    args = parser.parse_args()

    run_inference(args)