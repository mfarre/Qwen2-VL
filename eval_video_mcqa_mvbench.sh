set -x

EVAL_DATA_DIR=eval_data
OUTPUT_DIR=eval_output
CKPT_NAME="Qwen/Qwen2-VL-7B-Instruct"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/mvbench/answers/${CKPT_NAME}/merge.json

# judge if the number of json lines is 0
if [ ! -f "$output_file" ] || [ $(cat "$output_file" | wc -l) -eq 0 ]; then
    rm -f ${OUTPUT_DIR}/mvbench/answers/${CKPT_NAME}/*.json
fi

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 eval/inference_video_mcqa_mvbench.py \
            --model-path ${CKPT_NAME} \
            --video-folder ${EVAL_DATA_DIR}/mvbench/video \
            --question-file ${EVAL_DATA_DIR}/mvbench/json \
            --answer-file ${OUTPUT_DIR}/mvbench/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/mvbench/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi

python3 eval/eval_video_mcqa_mvbench.py \
    --pred_path ${output_file} \
