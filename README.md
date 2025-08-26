# Træning med 8 gpuer

1. `pip install -r requirements.txt`

2. `hf auth login`

3. `wandb login`

4. Gør nedenstående. Ret `nproc_per_node` alt efter hvor mange gpuer du har. Hvis du har 1 så slet NPROC og deepspeed.

```bash
# Configuration
nproc_per_node=8

# Run training with Swift SFT
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen3-Embedding-0.6B \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type full \
    --dataset syvai/giga-embed \
    --split_dataset_ratio 0.01 \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 15000 \
    --num_train_epochs 5 \
    --save_steps 15000 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --use_hf true \
    --report_to wandb \
    --deepspeed zero3
```