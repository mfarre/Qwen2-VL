import os
import re
import math
import json
import argparse
import warnings
import traceback
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class PerceptionTestMCQADataset(Dataset):
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_list, video_folder):
        self.data_list = data_list
        self.video_folder = video_folder

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_name = line['metadata']['video_id']
        mc_questions = line['mc_question']

        for fmt in self.video_formats:
            temp_path = os.path.join(self.video_folder, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break
        
        # video_input = {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0}
        video_input = {"type": "video", "video": video_path, "fps": 1.0}

        instructs = []
        qids = []
        ops = []
        for q in mc_questions:
            question = q['question']
            qid = q['id']
            options = q['options']
            instruct = f'Question: {question}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'

            instructs.append(instruct)
            qids.append(qid)
            ops.append(options)

        return {
            'video': video_input,
            'video_id': video_name,
            'instructs': instructs,
            'question_ids': qids,
            'options': ops,
        }

def collate_fn(batch):
    vid = [x['video'] for x in batch]
    v_id = [x['video_id'] for x in batch]
    ins = [x['instructs'] for x in batch]
    q_ids = [x['question_ids'] for x in batch]
    ops = [x['options'] for x in batch]
    return vid, v_id, ins, q_ids, ops

def run_inference(args):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    questions = json.load(open(args.question_file, "r"))
    questions = list(questions.values())
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    assert args.batch_size == 1, "Batch size must be 1 for inference"
    dataset = PerceptionTestMCQADataset(questions, args.video_folder)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    for i, (video_input, video_id, instructs, question_ids, options) in enumerate(tqdm(dataloader)):
        video_input = video_input[0]
        video_id = video_id[0]
        instructs = instructs[0]
        question_ids = question_ids[0]
        options = options[0]

        qas = []
        for idx, instruct in enumerate(instructs):
            letters = ['(A)', '(B)', '(C)']
            question_id = question_ids[idx]
            _options = options[idx]

            messages = [
                {
                    "role": "user",
                    "content": [
                        video_input,
                        {"type": "text", "text": instruct},
                    ],
                }
            ]

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

            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            pred_answer = re.findall('\(*[A-C]\)*', output)
            try:
                assert len(pred_answer) >= 1, f'The video "{video_id}" output "{output}" is not in the expected format'
                pred_answer = pred_answer[0].strip()
                pred_answer = pred_answer.strip('()')
                pred_answer = f'({pred_answer})'
                pred_idx = letters.index(pred_answer)
            except:
                traceback.print_exc()
                tmp_options = [x.lower() for x in _options]
                if output.lower() in tmp_options:
                    tmp_options = [x.lower() for x in _options]
                    pred_idx = tmp_options.index(output.lower())
                else:
                    pred_idx = 2

            qas.append({'id': question_id, 'answer_id': pred_idx, 'answer': _options[pred_idx]})

        ans_file.write(f'"{video_id}": {json.dumps(qas)},\n')

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
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    args = parser.parse_args()

    run_inference(args)