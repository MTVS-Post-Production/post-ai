import os, sys, shutil

flask_dir = os.getcwd() + "\\pypost"
sys.path.append(flask_dir)

import base64
import cv2
from flask import Blueprint, request, jsonify
import math
import torch
from video_classification import PoseClassification, make_clip
import time

bp = Blueprint('checkpose', __name__, url_prefix='/checkpose')

# mp4 파일을 받아 pose estimation을 진행
@bp.route('/', methods=['GET', 'POST'])
def video_pose():
    starttime = time.time()
    print()
    pose = request.get_json()['pose']  # 스프링 서버에서 전달 받은 base64로 인코딩된 파일
    pose_video = base64.b64decode(pose)

    # 디코딩한 mp4 파일을 폴더에 저장
    os.makedirs('./pypost/pose_estimation', exist_ok=True)
    check_video = './pypost/pose_estimation/received_file.mp4'
    with open(check_video, 'wb') as file:
        file.write(pose_video)

    # 만약 영상의 크기가 6kb보다 작으면 오류 영상이라고 판단해 아무것도 리턴하지 않음
    file_byte = os.path.getsize(check_video)
    file_kbyte = file_byte // 1024

    if file_kbyte < 6:
        print("Error: There's no file to check")
        response_data = []
        return jsonify(response_data)

    # 만약 길이가 20초 이상이면 비디오 분할 수행
    cap = cv2.VideoCapture(check_video)
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_duration = math.floor(frame_length / fps)
    cap.release()
    cv2.destroyAllWindows()

    if video_duration > 20:
        make_clip(check_video)
        # 기존 영상은 bin 폴더를 만들어 이동시켜 원본 영상을 확인할 수 있게끔 조치
        os.makedirs('./pypost/bin', exist_ok=True)
        shutil.move(check_video, './pypost/bin/received_file.mp4')

    torch.cuda.empty_cache()
    model_ckpt = './checkpoint-7566'  # 모델 ckpt 불러오기
    video_classification = PoseClassification(model_ckpt)

    # pose 추론 진행
    video_list = os.listdir('./pypost/pose_estimation')
    response_data = []
    for i in range(len(video_list)):
        video = f'./pypost/pose_estimation/{video_list[i]}'
        video_classification.load_video(video)
        result = video_classification.predict()
        if result == 'run':
            result = 'walk'
        response_data.append(result)
    
    endtime = time.time()
    print(f"응답 전송:{response_data}, 소요 시간:{endtime-starttime}s")

    return jsonify(response_data)