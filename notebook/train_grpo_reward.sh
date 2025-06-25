CDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
swift rlhf \
    --rlhf_type grpo \
    --model /mnt/workspace/model/Qwen3-0.6B \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs format \
    --reward_model /mnt/workspace/model/Qwen3-4B \
    --reward_model_plugin genrm \
    --reward_weights 0.2 1 \
    --vllm_gpu_memory_utilization 0.5 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --gc_collect_after_offload true \
    --log_completions true \
    --output_dir output2 \
    --deepspeed zero3