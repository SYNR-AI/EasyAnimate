source ~/.bashrc
source activate
conda activate /vepfs-zulution/shuiyunhao/conda/envs/hunyuanvideo
cd /cv/shuiyunhao/task/EasyAnimate

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export MODEL_NAME="/cv/models/EasyAnimateV5.1-7b-zh"
export DATASET_NAME=""
export DATASET_META_NAME="datasets/Minimalism/metadata_add_width_height.json"

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch \
  --use_deepspeed \
  --deepspeed_config_file config/zero_stage2_config.json \
  --deepspeed_hostfile $MLP_MPI_HOSTFILE \
  --num_processes=$(($MLP_WORKER_GPU * $MLP_WORKER_NUM)) \
  --num_machines=$MLP_WORKER_NUM \
  --machine_rank=$MLP_ROLE_INDEX \
  --main_process_port $MLP_WORKER_0_PORT \
  --main_process_ip $MLP_WORKER_0_HOST \
  scripts/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_v5.1_magvit_qwen.yaml" \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=1 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=5e-3 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --use_deepspeed \
  --train_mode="normal" \
  --trainable_modules "." \
  --loss_type="flow"