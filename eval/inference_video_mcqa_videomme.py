import os
import re
import math
import json
import copy
import argparse
import warnings
import traceback

import torch
import pysubs2
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Set the PyTorch memory management option
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class VideoMMEDataset(Dataset):
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, video_folder, subtitle_folder, data_list, processor):
        self.video_folder = video_folder
        self.subtitle_folder = subtitle_folder
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_ytid = line['url'].split('watch?v=')[-1]

        video_path = None
        for fmt in self.video_formats:
            temp_path = os.path.join(self.video_folder, f'{video_ytid}{fmt}')
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        subtitle_path = os.path.join(self.subtitle_folder, f'{video_ytid}.srt')

        try:
            video_tensor = {"type": "video", "video": video_path, "fps": 1.0}
        except:
            traceback.print_exc()
            print(f'It occurs error when reading {video_ytid}')
            video_tensor = None

        if video_tensor is not None and os.path.exists(subtitle_path):
            subs = pysubs2.load(subtitle_path, encoding="utf-8")
            subtitles = [sub.text.replace("\\N", " ") for sub in subs]
            subtitles = "\n".join(subtitles)
        else:
            subtitles = ""

        return {
            'video': video_tensor,
            'subtitle': subtitles,
            'record': line,
        }

def collate_fn(batch):
    vid = [x['video'] for x in batch]
    sub = [x['subtitle'] for x in batch]
    rcs = [x['record'] for x in batch]
    return vid, sub, rcs

def load_parquet(parquet_file):
    table = pq.read_table(parquet_file)
    df = table.to_pandas()

    jsons = []
    for record in df.itertuples():
        if len(jsons) < int(record.video_id):
            jsons.append({
                "video_id": record.video_id,
                "youtube_id": record.videoID,
                "url": record.url,
                "duration": record.duration,
                "domain": record.domain,
                "sub_category": record.sub_category,
                "questions": [
                    {
                        "question_id": record.question_id,
                        "task_type": record.task_type,
                        "question": record.question,
                        "choices": list(record.options),
                        "answer": record.answer,
                    }
                ]
            })
        else:
            jsons[-1]['questions'].append({
                "question_id": record.question_id,
                "task_type": record.task_type,
                "question": record.question,
                "choices": list(record.options),
                "answer": record.answer,
            })

    return jsons

def build_videomme_eval(args, processor):
    questions = load_parquet(args.question_file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = VideoMMEDataset(args.video_folder, args.subtitle_folder, questions, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return dataloader

def videomme_dump(record, instruct, output):
    letters = ['A', 'B', 'C', 'D']
    pred_answer = re.findall('[\(\ \[]*([A-D])[\)\.\ \]]*', output)
    try:
        assert len(pred_answer) >= 1, 'The video \"{}\" output \"{}\" is not in the expected format'.format(record['youtube_id'], instruct + '\n' + output)
        pred_answer = pred_answer[0].strip()
        pred_answer = pred_answer.strip('()')
        pred_idx = letters.index(pred_answer)
    except:
        traceback.print_exc()
        pred_idx = 2
    return letters[pred_idx]

def reduce_video_frames(video_input, max_frames=50):
    """
    Reduce the number of frames in a video tensor to a maximum of max_frames.
    
    Args:
    video_input (torch.Tensor): Input video tensor of shape [num_frames, channels, height, width]
    max_frames (int): Maximum number of frames in the output tensor
    
    Returns:
    torch.Tensor: Video tensor with reduced number of frames
    """
    num_frames, channels, height, width = video_input.shape
    
    if num_frames <= max_frames:
        return video_input
    
    # Calculate indices of frames to keep
    keep_indices = torch.linspace(0, num_frames - 1, max_frames).long()
    
    # Select frames
    reduced_video = video_input[keep_indices]
    
    return reduced_video
def run_inference(args):
    # Modify the configuration parameters
    max_length = 65536

    # 1. Load the original configuration
    config = AutoConfig.from_pretrained(args.model_path)
    
    # 2. Modify the configuration
    config.sliding_window = max_length
    config.max_position_embeddings = max_length
    config.model_max_length = max_length
    
    # 3. Initialize the model with the modified configuration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    
    # 4. Initialize and update the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.model_max_length = max_length
    
    # 5. Initialize and update the processor
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer = tokenizer  # Use the updated tokenizer
    
    # 6. Verify the configurations
    print(f"Model config - Sliding window: {model.config.sliding_window}")
    print(f"Model config - Max position embeddings: {model.config.max_position_embeddings}")
    print(f"Model config - Model max length: {model.config.model_max_length}")
    print(f"Tokenizer max length: {tokenizer.model_max_length}")
    print(f"Processor image size: {processor.image_processor.size}")
    print(f"Processor tokenizer max length: {processor.tokenizer.model_max_length}")

    answer_file = os.path.expanduser(args.answer_file)
    answer_sub_file = answer_file.replace('.json', '_sub.json')
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    ans_sub_file = open(answer_sub_file, "w")

    val_loader = build_videomme_eval(args, processor)

    for i, (videos, subtitles, records) in enumerate(tqdm(val_loader)):
        # Clear CUDA cache at the start of each video iteration
        torch.cuda.empty_cache()

        video = videos[0]
        subtitle = subtitles[0]
        record = records[0]

        new_record = copy.deepcopy(record)
        new_record_sub = copy.deepcopy(record)

        if video is None:
            new_record['missing'] = True
            ans_file.write(json.dumps(new_record) + ",\n")
            new_record_sub['missing'] = True
            ans_sub_file.write(json.dumps(new_record_sub) + ",\n")
            continue
        else:
            new_record['missing'] = False
            new_record_sub['missing'] = False

        questions = record['questions']
        for idx, question in enumerate(questions):
            q = question['question']
            ops = question['choices']

            instruct = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n"
            instruct += f"{q}\n"
            for op_idx, op in enumerate(ops):
                instruct += f"{op}\n"
            instruct += "The best answer is: "

            # Prepare messages for Qwen2-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        video,
                        {"type": "text", "text": instruct},
                    ],
                }
            ]

            # Process input for Qwen2-VL
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            reduced_video_inputs = [reduce_video_frames(vi, max_frames=32) for vi in video_inputs]

            for v in reduced_video_inputs:
                print(f"Type: {type(v)}, Shape: {v.shape if hasattr(v, 'shape') else 'N/A'}")

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=reduced_video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Generate output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            new_record['questions'][idx]['response'] = videomme_dump(record, instruct, output)

            # Process with subtitle
            instruct_with_sub = f"This video's subtitles are listed below:\n{subtitle}\n" + instruct
            messages_with_sub = [
                {
                    "role": "user",
                    "content": [
                        video,
                        {"type": "text", "text": instruct_with_sub},
                    ],
                }
            ]

            text_with_sub = processor.apply_chat_template(messages_with_sub, tokenize=False, add_generation_prompt=True)
            inputs_with_sub = processor(
                text=[text_with_sub],
                images=image_inputs,
                videos=reduced_video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs_with_sub = inputs_with_sub.to(model.device)

            generated_ids_with_sub = model.generate(**inputs_with_sub, max_new_tokens=128)
            generated_ids_trimmed_with_sub = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_with_sub.input_ids, generated_ids_with_sub)]
            output_with_sub = processor.batch_decode(generated_ids_trimmed_with_sub, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            new_record_sub['questions'][idx]['response'] = videomme_dump(record, instruct_with_sub, output_with_sub[0])

        ans_file.write(json.dumps(new_record) + ",\n")
        ans_sub_file.write(json.dumps(new_record_sub) + ",\n")

    ans_file.close()
    ans_sub_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='Path to the Qwen2-VL model', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--subtitle-folder', help='Directory containing subtitle files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    run_inference(args)