import json
from collections import defaultdict
import sys

#python eval_video_mcqa_mvbench_results.py eval_output/.../merge.json

# Load the benchmark results from the JSON file
with open(sys.argv[1], 'r') as f:
    results = [json.loads(line) for line in f]

# Initialize variables to track correct answers
correct_count = 0
total_count = 0

# Initialize a dictionary to track correct answers per task type
task_type_correct = defaultdict(int)
task_type_total = defaultdict(int)

# Process each entry in the results
for entry in results:
    pred = entry['pred']
    gt = entry['gt']
    task_type = entry['task_type']
    
    # Update overall counts
    total_count += 1
    if pred == gt:
        correct_count += 1
    
    # Update per task type counts
    task_type_total[task_type] += 1
    if pred == gt:
        task_type_correct[task_type] += 1

# Calculate the overall percentage of correct answers
overall_percentage_correct = (correct_count / total_count) * 100

# Calculate the percentage of correct answers per task type
task_type_percentage_correct = {
    task: (correct / task_type_total[task]) * 100
    for task, correct in task_type_correct.items()
}

# Print the results
print(f"Overall Percentage of Correct Answers: {overall_percentage_correct:.2f}%")

print("\nPercentage of Correct Answers per Task Type:")
for task_type, percentage in task_type_percentage_correct.items():
    print(f"{task_type}: {percentage:.2f}%")
