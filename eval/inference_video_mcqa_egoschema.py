import os
import re
import math
import json
import argparse
import warnings
import traceback

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
class EgoschemaDataset(Dataset):
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_folder, data_list):
        self.data_folder = data_folder
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        q_uid = line['q_uid']  # Ensure this is correctly accessed from the data

        for fmt in self.video_formats:
            temp_path = os.path.join(self.data_folder, f"{q_uid}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        video_input = {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0}

        question = line['question']
        a0, a1, a2, a3, a4 = line['option 0'], line['option 1'], line['option 2'], line['option 3'], line['option 4']
        instruct = f'Question: {question}\nOptions:\n(A) {a0}\n(B) {a1}\n(C) {a2}\n(D) {a3}\n(E) {a4}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'

        return {
            'q_uid': q_uid,
            'video': video_input, 
            'instruct': instruct,
        }

def build_egoschema_eval(args):
    questions = json.load(open(args.question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = EgoschemaDataset(args.video_folder, questions)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return dataloader

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

    val_loader = build_egoschema_eval(args)

    for batch in tqdm(val_loader):
        q_uid = batch['q_uid'][0] if isinstance(batch['q_uid'], list) else batch['q_uid']
        instruct = batch['instruct'][0] if isinstance(batch['instruct'], list) else batch['instruct']

        # Note: We're not using 'video' key as it's not present in the batch
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruct},
                ],
            }
        ]

        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # Prepare a dictionary for egoschema_dump
            result = {
                'q_uid': q_uid,
                'instruct': instruct
            }
            egoschema_dump(ans_file, result, output)
        except Exception as e:
            print(f"Error processing q_uid {q_uid}: {str(e)}")
            # Write a default answer or skip this question
            ans_file.write(f'{q_uid}, -1\n')

    ans_file.close()

def egoschema_dump(ans_file, line, output):
    q_uid = line['q_uid']
    letters = ['A', 'B', 'C', 'D', 'E']

    pred_answer = re.findall('[\(\ ]*[A-E][\)\ ]*', output)
    try:
        if len(pred_answer) >= 1:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
        else:
            print(f'The video "{q_uid}" output "{output}" is not in the expected format')
            pred_idx = -1  # or some default value
    except Exception as e:
        print(f"Error processing output for q_uid {q_uid}: {str(e)}")
        pred_idx = -1  # or some default value

    ans_file.write(f'{q_uid}, {pred_idx}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiple-Choice Video QA Evaluation Script for Qwen2-VL.')
    parser.add_argument('--model-path', help='Path to the Qwen2-VL model', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    run_inference(args)