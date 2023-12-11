from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
import torch
from torchvision.io import read_video
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToPILImage, ToTensor
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import cv2
import os
import math

# pose 분류
class PoseClassification():
    def __init__(self, model_ckpt):
        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            ignore_mismatched_sizes=True)

        self.mean = self.image_processor.image_mean
        self.std = self.image_processor.image_std
        if "shortest_edge" in self.image_processor.size:
            height = width = self.image_processor.size["shortest_edge"]
        else:
            height = self.image_processor.size["height"]
            width = self.image_processor.size["width"]
        self.resize_to = (height, width)
        self.num_frames_to_sample = self.model.config.num_frames

    # 비디오 전처리
    def load_video(self, video_file):
        self.video_file = video_file
        self.video_file_path = '/'.join(video_file.split('/')[:-1])
        self.video_file_name = video_file.split('/')[-1].split('.')[0]
    
        video, _, _ = read_video(video_file, pts_unit="sec")
        self.video = video.permute(3, 0, 1, 2)

        return self.video

    def transform_video(self, video):
        video = video.permute(1, 0, 2, 3)
        transformed_video = []
        for frame in video:
            frame = frame / 255.0
            frame = ToPILImage()(frame)
            frame = Resize(self.resize_to, antialias=True)(frame)
            frame = ToTensor()(frame)
            frame = Normalize(self.mean, self.std)(frame)
            transformed_video.append(frame)
        transformed_video = torch.stack(transformed_video)

        return transformed_video.permute(1, 0, 2, 3)

    def preprocessing(self):
        val_transform = Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(self.transform_video),
                        ]
                    ),
                ),
            ])
        self.video_tensor = val_transform({"video": self.video})["video"]

        return self.video_tensor

    def run_inference(self, model, video):
        perumuted_sample_test_video = video.permute(1, 0, 2, 3)
        inputs = {"pixel_values": perumuted_sample_test_video.unsqueeze(0)}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model = model.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        return logits

    # 추론 실행
    def predict(self):
        self.preprocessing()
        logits = self.run_inference(self.model, self.video_tensor)
        predicted_class_idx = logits.argmax(-1).item()

        return self.model.config.id2label[predicted_class_idx]


# 20초 단위로 비디오 분할
def make_clip(video_path):
    # Video Status
    cap = cv2.VideoCapture(video_path)

    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_duration = math.floor(frame_length / fps)

    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    if video_duration > 20:
        start_frame = 0
        video_index = 1

        cut_duration = 20
        cut_frame = int(cut_duration * fps)
        overlap_duration = 10
        overlap_frame = int(overlap_duration * fps)

        while start_frame + cut_frame <= frame_length:
            end_frame = start_frame + cut_frame
            clip_video_path = f'./pypost/pose_estimation/clip_video_{video_index}.mp4'
            output = cv2.VideoWriter(clip_video_path, fourcc=fourcc, fps=fps, frameSize=(int(video_width), int(video_height)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                success, image = cap.read()
                if not success:
                    break
                output.write(image)

            start_frame += cut_frame - overlap_frame
            video_index += 1

            output.release()

        cap.release()
        cv2.destroyAllWindows()

        print("done")

        return "done"

    else:
        cap.release()
        cv2.destroyAllWindows()
        print("영상을 분할할 필요가 없습니다.")

        return "Not done"


# 영상을 1:1 비율에 맞춰서 자르기
def VideoCrop(video_path):
    cap = cv2.VideoCapture(video_path)
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = './pypost/pose_estimation/cropped_file.mp4'

    # 높이가 더 크다면 3:1의 비율로 자르기
    if video_height > video_width:
        margin = (video_height - video_width)
        new_height = video_height - margin
        start_y = int(margin / 4 * 3)

        output = cv2.VideoWriter(output_path, fourcc=fourcc, fps=fps, frameSize=(int(video_width), int(new_height)))

        while True:
            success, frame = cap.read()
            if not success:
                break

            cropped_frame = frame[start_y:start_y+new_height, :]
            output.write(cropped_frame)

        output.release()
        print("crop done")
        cap.release()
        cv2.destroyAllWindows()

    elif video_height < video_width:
        margin = (video_width - video_height)
        new_width = video_width - margin
        start_x = int(margin / 2)

        output = cv2.VideoWriter(output_path, fourcc=fourcc, fps=fps, frameSize=(int(new_width), int(video_height)))

        while True:
            success, frame = cap.read()
            if not success:
                break

            cropped_frame = frame[:, start_x:start_x+new_width]
            output.write(cropped_frame)

        output.release()
        print("crop done")
        cap.release()
        cv2.destroyAllWindows()

    else:
        cap.release()
        cv2.destroyAllWindows()
        print("영상을 crop할 필요가 없습니다.")
        os.rename(video_path, output_path)

    return output_path