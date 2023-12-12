import os, sys, shutil
from textual_inversion import textual_inversion as TI
import preprocess_image as PI
import torch
import subprocess
from glob import glob

class Make_3d_Modeling():
    def __init__(self, filepath, is_bool, token):
        self.filepath = filepath
        self.device = 'cuda'
        self.is_bool = is_bool
        self.token = token
        self.python_cmd = '/home/meta-ai2/anaconda3/envs/Magic123/bin/python'

    # Step 1: 이미지 전처리
    def preprocess(self):
        # 결과물 폴더 생성
        self.new_folder = self.filepath.split('/')[-1].split('.')[0]  # ex) ./run/mask.jpg >> mask
        print('self.new_folder: ', self.new_folder)
        self.new_path = f'./run/{self.new_folder}'
        print('self.new_path: ', self.new_path)
        os.makedirs(self.new_path + '/images', exist_ok=True)
        shutil.copyfile(self.filepath, self.new_path + '/images/image.jpg')

        self.filepath = self.new_path + '/images/image.jpg'  # 이미지 경로를 파일 경로로 업데이트
        self.depth_estimator = PI.DepthEstimator()
        PI.process_single_image(self.filepath, self.depth_estimator)

        return "preprocess finished!"

    def textual_inversion(self):
        torch.cuda.empty_cache()
        if self.is_bool==True:
            self.pretrained_model = 'runwayml/stable-diffusion-v1-5'
            self.train_data_dir = self.new_path + '/images/rgba.png'
            self.learnable_property = 'object'
            self.placeholder_token = f'_{self.new_folder}_'
            self.initializer_token = f'{self.token}'
            self.resolution = 512
            self.train_batch_size = 10
            self.gradient_accumulation_steps = 1
            self.max_train_steps = 3000
            self.lr_scheduler = 'constant'
            self.lr_warmup_steps = 0
            os.makedirs(self.new_path + '/textual_inversion', exist_ok=True)
            self.output_dir = self.new_path + '/textual_inversion'

            cmd = self.python_cmd + ' textual_inversion/textual_inversion.py -O' \
                 + f' --pretrained_model_name_or_path "{self.pretrained_model}" \
                      --train_data_dir {self.train_data_dir} \
                      --learnable_property {self.learnable_property} \
                      --placeholder_token {self.placeholder_token} \
                      --initializer_token {self.initializer_token} \
                      --resolution {self.resolution} \
                      --train_batch_size {self.train_batch_size} \
                      --gradient_accumulation_steps {self.gradient_accumulation_steps} \
                      --max_train_steps {self.max_train_steps} \
                      --lr_scheduler {self.lr_scheduler} \
                      --lr_warmup_steps {self.lr_warmup_steps} \
                      --output_dir {self.output_dir} \
                      --use_augmentations'
            
            subprocess.run(cmd, shell=True)
            shutil.move(self.output_dir + '/learned_embeds.bin', self.new_path + '/images/learned_embeds.bin')
        else:
            pass

    def first_progress(self):
        torch.cuda.empty_cache()
        self.sd_version = '1.5'
        self.image = self.new_path + '/images/rgba.png'
        self.first_workspace = self.new_path + '/outputs/first'
        self.optim = 'adam'
        self.iters = 5000
        self.guidance = "SD zero123"
        self.lambda_guidance_1 = 1.0
        self.lambda_guidance_2 = 40.0
        self.guidance_scale_1 = 100.0
        self.guidance_scale_2 = 5.0
        self.latent_iter_ratio = 0.0
        self.normal_iter_ratio = 0.2
        self.t_range_1 = 0.2
        self.t_range_2 = 0.6
        self.bg_radius = -1.0
        self.density_activation = "relu"
        
        if self.is_bool==True:
            self.text = "A high-resolution DSLR image of <token>"
            self.learned_embeds_path = self.new_path + '/images/learned_embeds.bin'
            cmd = self.python_cmd + ' main.py -O' \
                 + f' --text "{self.text}" \
                      --sd_version {self.sd_version} \
                      --image {self.image} \
                      --learned_embeds_path {self.learned_embeds_path} \
                      --workspace {self.first_workspace} \
                      --optim {self.optim} \
                      --iters {self.iters} \
                      --guidance {self.guidance} \
                      --lambda_guidance {self.lambda_guidance_1} {self.lambda_guidance_2} \
                      --guidance_scale {self.guidance_scale_1} {self.guidance_scale_2} \
                      --latent_iter_ratio {self.latent_iter_ratio} \
                      --normal_iter_ratio {self.normal_iter_ratio} \
                      --t_range {self.t_range_1} {self.t_range_2} \
                      --bg_radius {self.bg_radius} \
                      --save_mesh'

            subprocess.run(cmd, shell=True)
            
        else:
            self.text = f"A high-resolution DSLR image of a {self.new_folder}"
            # subprocess로 스크립트 내부에서 .py 파일 실행 / flask 실행 전 --fp16 --vram_O 확인!
            cmd = self.python_cmd + ' main.py -O' \
                 + f' --text "{self.text}" \
                      --sd_version {self.sd_version} \
                      --image {self.image} \
                      --workspace {self.first_workspace} \
                      --optim {self.optim} \
                      --iters {self.iters} \
                      --guidance {self.guidance} \
                      --lambda_guidance {self.lambda_guidance_1} {self.lambda_guidance_2} \
                      --guidance_scale {self.guidance_scale_1} {self.guidance_scale_2} \
                      --latent_iter_ratio {self.latent_iter_ratio} \
                      --normal_iter_ratio {self.normal_iter_ratio} \
                      --t_range {self.t_range_1} {self.t_range_2} \
                      --bg_radius {self.bg_radius} \
                      --density_activation {self.density_activation} \
                      --save_mesh'

            subprocess.run(cmd, shell=True)

        return "first_progress finished!"

    def second_progress(self):
        torch.cuda.empty_cache()
        self.second_workspace = self.new_path + '/outputs/retouch'
        self.checkpoint = self.first_workspace + '/checkpoints/first.pth'
        self.known_view_interval = 4
        self.lambda_guidance_1 = 1e-3
        self.lambda_guidance_2 = 0.01

        if self.is_bool==True:
            self.text = "A high-resolution DSLR image of <token>"
            self.learned_embeds_path = self.new_path + '/images/learned_embeds.bin'
            cmd = self.python_cmd + ' main.py -O' \
                 + f' --text "{self.text}" \
                      --sd_version {self.sd_version} \
                      --image {self.image} \
                      --learned_embeds_path {self.learned_embeds_path} \
                      --workspace {self.second_workspace} \
                      --dmtet --init_ckpt {self.checkpoint} \
                      --optim {self.optim} \
                      --iters {self.iters} \
                      --known_view_interval {self.known_view_interval} \
                      --guidance {self.guidance} \
                      --lambda_guidance {self.lambda_guidance_1} {self.lambda_guidance_2} \
                      --guidance_scale {self.guidance_scale_1} {self.guidance_scale_2} \
                      --latent_iter_ratio {self.latent_iter_ratio} \
                      --rm_edge \
                      --bg_radius {self.bg_radius} \
                      --save_mesh'            
            
            subprocess.run(cmd, shell=True)

        else:
            self.text = f"A high-resolution DSLR image of a {self.new_folder}"
            cmd = self.python_cmd + ' main.py -O' \
                 + f' --text "{self.text}" \
                      --sd_version {self.sd_version} \
                      --image {self.image} \
                      --workspace {self.second_workspace} \
                      --dmtet --init_ckpt {self.checkpoint} \
                      --optim {self.optim} \
                      --iters {self.iters} \
                      --known_view_interval {self.known_view_interval} \
                      --guidance {self.guidance} \
                      --lambda_guidance {self.lambda_guidance_1} {self.lambda_guidance_2} \
                      --guidance_scale {self.guidance_scale_1} {self.guidance_scale_2} \
                      --latent_iter_ratio {self.latent_iter_ratio} \
                      --rm_edge \
                      --density_activation {self.density_activation} \
                      --bg_radius {self.bg_radius} \
                      --save_mesh'            
            
            subprocess.run(cmd, shell=True)

        return 'second_progress finished!'
    
    def run(self):
        self.preprocess()
        self.textual_inversion()
        self.first_progress()
        self.second_progress()

        print('Process Finished!!')