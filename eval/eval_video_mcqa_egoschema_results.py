import json
import csv
import sys


#python eval_video_mcqa_egoschema_results.py eval/egoschema_subset_answers.json eval_output/.../merge.csv
# Load the ground truth from the JSON file
with open(sys.argv[1], 'r') as f:
    ground_truth = json.load(f)

# Initialize variables to track correct answers
correct_count = 0
total_questions = len(ground_truth)

# Read the answers from the CSV file and compare with ground truth
with open(sys.argv[2], 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header if present
    for row in reader:
        q_uid = row[0]
        answer = int(row[1])
        # Check if the answer is in the ground truth and compare
        if q_uid in ground_truth and ground_truth[q_uid] == answer:
            correct_count += 1

# Calculate the percentage of correct answers
percentage_correct = (correct_count / total_questions) * 100

# Report the percentage of correct answers
print(f"Percentage of correct answers: {percentage_correct:.2f}%")