#!/bin/bash

#BASE_MODEL="deepseekr1/DeepSeek-R1-Distill-Llama-8B"
FINE_MODEL="deepseekr1/DeepSeek-R1-8B-FINETINED-v0-2"
#BASE_OUTPUT="deepseek_r1_BASE_eval"
FINE_OUTPUT="deepseek_r1_FINE_eval"
#TASKS="asdiv,gsm8k,hendrycks_math,mathqa,mgsm_en_cot_en,mgsm_direct_en,minerva_math,hrm8k"
TASKS="asdiv,gsm8k"
EXAMPLES=10

cd lm-evaluation-harness || exit

# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained="/home/joshua/llms/$BASE_MODEL",trust_remote_code=True,dtype=auto,load_in_4bit=True \
#     --tasks $TASKS \
#     --num_fewshot $EXAMPLES \
#     --device cuda:0 \
#     --batch_size auto \
#     --output_path ../$BASE_OUTPUT.json \
#     --trust_remote_code

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained="/home/joshua/llms/$FINE_MODEL",trust_remote_code=True,dtype=auto,load_in_4bit=True \
    --tasks $TASKS \
    --num_fewshot $EXAMPLES \
    --device cuda:0 \
    --batch_size auto \
    --output_path ../$FINE_OUTPUT.json \
    --trust_remote_code

cd ..

#--num_fewshot $EXAMPLES \