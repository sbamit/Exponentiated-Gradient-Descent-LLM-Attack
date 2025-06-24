#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llm-embedding-attack_env

# Fix shared library issues (GLIBCXX)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Common settings
BATCH_SIZE=10
NUM_STEPS=200

# List all 20 combinations
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Llama2 --dataset_name AdvBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Llama2 --dataset_name HarmBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Llama2 --dataset_name JailbreakBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Llama2 --dataset_name MaliciousInstruct --batch_size $BATCH_SIZE --num_steps $NUM_STEPS

# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Falcon --dataset_name AdvBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Falcon --dataset_name HarmBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Falcon --dataset_name JailbreakBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Falcon --dataset_name MaliciousInstruct --batch_size $BATCH_SIZE --num_steps $NUM_STEPS

# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name MPT --dataset_name AdvBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name MPT --dataset_name HarmBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name MPT --dataset_name JailbreakBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name MPT --dataset_name MaliciousInstruct --batch_size $BATCH_SIZE --num_steps $NUM_STEPS

# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Vicuna --dataset_name AdvBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Vicuna --dataset_name HarmBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Vicuna --dataset_name JailbreakBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Vicuna --dataset_name MaliciousInstruct --batch_size $BATCH_SIZE --num_steps $NUM_STEPS

# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Mistral --dataset_name AdvBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Mistral --dataset_name HarmBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Mistral --dataset_name JailbreakBench --batch_size $BATCH_SIZE --num_steps $NUM_STEPS
# python Run_multi-prompt_attack-llm_using_EGD_with_Adam_Optim.py --model_name Mistral --dataset_name MaliciousInstruct --batch_size $BATCH_SIZE --num_steps $NUM_STEPS

echo "All model–dataset jobs completed."
