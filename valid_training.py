import os
import time
import argparse
from easyanimate.ui.ui import EasyAnimateController
import torch
import shutil

def main(output_dir, skip_initial_valid):
    gpu_memory_mode = "model_cpu_offload_and_qfloat8"
    enable_teacache = True
    teacache_threshold = 0.08
    weight_dtype = torch.bfloat16

    controller = EasyAnimateController(gpu_memory_mode, enable_teacache, teacache_threshold, weight_dtype)

    controller.update_edition("v5")
    controller.update_diffusion_transformer("models/Diffusion_Transformer/EasyAnimateV5-12b-zh-InP")

    # controller.update_edition("v5.1")
    # controller.update_diffusion_transformer("models/Diffusion_Transformer/EasyAnimateV5.1-12b-zh-InP")

    # 定义扫描间隔时间
    scan_interval = 60  # 1分钟

    # 当前处理到的最大iter
    current_iter = 0

    # 初始化时获取最新的checkpoint数
    if skip_initial_valid:
        for folder in os.listdir(output_dir):
            if folder.startswith("checkpoint-"):
                iter_num = int(folder.split('-')[1])
                current_iter = max(current_iter, iter_num)

    while True:
        # 找到output_dir下的最大checkpoint文件夹
        max_iter = current_iter
        max_checkpoint_path = None
        for folder in os.listdir(output_dir):
            if folder.startswith("checkpoint-"):
                iter_num = int(folder.split('-')[1])
                if iter_num > max_iter:
                    max_iter = iter_num
                    max_checkpoint_path = os.path.join(output_dir, folder)

        # 如果找到新的最大checkpoint，进行验证
        if max_checkpoint_path and max_iter > current_iter:
            # 更新base model
            base_model_dropdown = os.path.join(max_checkpoint_path, "transformer/diffusion_pytorch_model.safetensors")
            print(base_model_dropdown)
            controller.update_base_model(base_model_dropdown)

            # 遍历valid_data文件夹
            for file in os.listdir('./valid_data'):
                if file.endswith('.png') or file.endswith('.jpg'):
                    start_image = os.path.join('./valid_data', file)
                    prompt_file = start_image.replace('.png', '.txt')
                    if os.path.exists(prompt_file):
                        with open(prompt_file, 'r') as f:
                            prompt = f.read().strip()
                            
                        valid_folder = os.path.join(max_checkpoint_path, "valid")
                        os.makedirs(valid_folder, exist_ok=True)

                        # 生成视频
                        for base_resolution in [512, 768, 960]:
                            destination_path = os.path.join(valid_folder, file.replace('.png', f'_{base_resolution}.mp4').replace('.jpg', f'_{base_resolution}.mp4'))
                            if os.path.exists(destination_path):
                                print(f"skip {file}_{base_resolution} because it already exists")
                                continue
                            save_sample_path, comment = controller.generate(
                                diffusion_transformer_dropdown="models/Diffusion_Transformer/EasyAnimateV5.1-12b-zh-InP",
                                motion_module_dropdown='none',
                                base_model_dropdown=base_model_dropdown,
                                lora_model_dropdown='none',
                                lora_alpha_slider=0.55,
                                prompt_textbox=prompt,
                                negative_prompt_textbox="Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code.",
                                sampler_dropdown="Flow",
                                sample_step_slider=50,
                                resize_method="Resize according to Reference",
                                width_slider=0,
                                height_slider=0,
                                base_resolution=base_resolution,
                                generation_method="Video Generation",
                                length_slider=49,
                                overlap_video_length=4,
                                partial_video_length=25,
                                cfg_scale_slider=6,
                                start_image=start_image,
                                end_image=None,
                                validation_video=None,
                                validation_video_mask=None,
                                control_video=None,
                                denoise_strength=0.70,
                                seed_textbox=43,
                                is_api=True,
                            )

                            shutil.copy(save_sample_path, destination_path)
                            print(f"saved to {destination_path}")

            # 更新current_iter为最新的checkpoint数
            current_iter = max_iter

        # 等待一段时间后再次扫描
        time.sleep(scan_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run validation on new checkpoints.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing checkpoint folders.")
    parser.add_argument("--skip", action='store_true', help="Skip validation of the latest checkpoint on startup.")
    args = parser.parse_args()
    main(args.output_dir, args.skip)