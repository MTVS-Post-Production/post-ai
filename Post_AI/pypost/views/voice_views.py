import os, sys
import time

flask_dir = os.getcwd() + "\\pypost"
voice_dir = os.getcwd() + "\\pypost\\trans_voice"
sys.path.append(flask_dir)
sys.path.append(voice_dir)

from flask import Blueprint, request, render_template, jsonify
from pypost.trans_voice import AI_Convert
import base64
from google.cloud import storage

# Google Storage 등록
KEY_PATH = "D:/User/user/post-ai/Post_AI/pypost/google-storage.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= KEY_PATH

bp = Blueprint('voice', __name__, url_prefix='/voice')

@bp.route('/')
def voice_main():
    return "voice convert입니다."

# .wav, .mp3 파일 등을 받아서 음성 변조 작업을 진행
@bp.route('/convert', methods=['GET', 'POST'])
def convert_voice():
    starttime = time.time()
    print()
    voice = request.get_json()['voice']  # 스프링 서버에서 전달 받은 base64로 인코딩된 파일
    model_name = request.get_json()['model_name']  # 변환하려는 목소리 모델의 이름
    user_id = request.get_json()['user_id']  # user의 번호

    sid = model_name + ".pth"
    vc_transform = 0  # 옥타브
    index_rate = 0.5  # 변환 강도

    print("수신 완료")

    # 같은 user id의 파일 존재 시 버킷 내 객체 삭제
    bucket_name = 'voice_production'
    destination_blob_name = f'result_voice_{user_id}'

    voice_file = base64.b64decode(voice)
    input_audio = "./pypost/trans_voice/spring/received_file.wav"
    with open(input_audio, 'wb') as file:
        file.write(voice_file)

    # 만약 영상의 크기가 6kb보다 작으면 오류 음성이라고 판단해 이전에 저장된 파일을 리턴
    file_byte = os.path.getsize('./pypost/trans_voice/spring/received_file.wav')
    file_kbyte = file_byte // 1024

    if file_kbyte < 6:
        print("There's no file to convert")
        url = f"https://storage.cloud.google.com/voice_production/{destination_blob_name}"
        return jsonify(url)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    try:
        blob.reload()
        generation_match_precondition = blob.generation
        blob.delete(if_generation_match=generation_match_precondition)
        print(f"Blob {destination_blob_name} deleted.")
    except:
        print(f"Blob {destination_blob_name} does not exists.")
        pass

    time.sleep(0.5)

    file_index = f"./Post_AI/pypost/trans_voice/logs/{model_name}/added_IVF1203_Flat_nprobe_1_{model_name}_v2.index"

    convert_path = AI_Convert.Voice_Convert(sid, vc_transform, input_audio, file_index, index_rate)

    # 객체 업로드
    time.sleep(0.5)
    blob.upload_from_filename(convert_path)
    url = f"https://storage.cloud.google.com/voice_production/{destination_blob_name}"
    endtime = time.time()
    
    print(f"구글 스토리지 업로드 완료!! 소요 시간:{endtime-starttime}s")

    return jsonify(url)