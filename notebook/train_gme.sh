
# losses: plugin/loss.py
# 8*40G
MAX_PIXELS=1003520 \
swift sft \
    --model G:\code\pretrain_model_dir\_modelscope\gme-Qwen2-VL-2B-Instruct \
    --train_type lora \
    --dataset 'swift/TextCaps:emb' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --eval_strategy steps \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --lazy_tokenize true \
    --warmup_ratio 0.05 \
    --learning_rate 5e-6 \
    --deepspeed zero3 \
    --dataloader_num_workers 4 \
    --task_type embedding \
    --loss_type infonce \
    --dataloader_drop_last true
