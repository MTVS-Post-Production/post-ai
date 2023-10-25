import os, sys

flask_dir = os.getcwd() + "\\pypost"
voice_dir = os.getcwd() + "\\pypost\\trans_voice"
sys.path.append(flask_dir)
sys.path.append(voice_dir)

from flask import Blueprint, request, render_template, jsonify
from pypost.trans_voice import AI_Convert
import base64
from google.cloud import storage

# Google Storage에 저장된 voice의 길이를 가져옴
def list_blobs(bucket_name):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)

    blob_length = [blob.name for blob in blobs]

    return len(blob_length)

# Google Storage 등록
KEY_PATH = "D:/User/user/post-ai/Post_AI/pypost/google-storage.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= KEY_PATH

storage_client = storage.Client()
buckets = list(storage_client.list_buckets())

bp = Blueprint('voice', __name__, url_prefix='/voice')

@bp.route('/')
def voice_main():
    return "voice convert입니다."

# .wav, .mp3 파일 등을 받아서 음성 변조 작업을 진행
@bp.route('/convert', methods=['GET', 'POST'])
def convert_voice():
    voice = request.get_json()['voice']  # 스프링 서버에서 전달 받은 base64로 인코딩된 파일
    model_name = request.get_json()['model_name']  # 변환하려는 목소리 모델의 이름
    user_id = request.get_json()['user_id']  # user의 번호

    sid = model_name + ".pth"
    vc_transform = 0  # 옥타브
    index_rate = 0.7  # 변환 강도

    voice_file = base64.b64decode(voice)
    input_audio = "./pypost/trans_voice/spring/received_file.wav"
    with open(input_audio, 'wb') as file:
        file.write(voice_file)

    file_index = f"./Post_AI/pypost/trans_voice/logs/{model_name}/added_IVF1203_Flat_nprobe_1_{model_name}_v2.index"

    convert_path = AI_Convert.Voice_Convert(sid, vc_transform, input_audio, file_index, index_rate)

    bucket_name = 'voice_production'
    length = list_blobs(bucket_name)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    destination_blob_name = f'result_voice_{length}'
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(convert_path)

    url = f"https://storage.cloud.google.com/voice_production/{destination_blob_name}"

    return jsonify(url)