import os, sys

flask_dir = os.getcwd() + "\\pypost"
voice_dir = os.getcwd() + "\\pypost\\trans_voice"
sys.path.append(flask_dir)
sys.path.append(voice_dir)

from flask import Blueprint, request, render_template, jsonify
from pypost.trans_voice import AI_Convert
import base64

bp = Blueprint('voice', __name__, url_prefix='/voice')

@bp.route('/')
def voice_main():
    return "voice convert입니다."

# .wav, .mp3 파일 등을 받아서 음성 변조 작업을 진행
@bp.route('/convert', methods=['GET', 'POST'])
def convert_voice():
    voice = request.get_json()['voice']  # 스프링 서버에서 전달 받은 base64로 인코딩된 파일
    model_name = request.get_json()['model_name']
    sid = model_name + ".pth"
    vc_transform = 0  # 옥타브
    index_rate = 0.7  # 변환 강도
    file_format = request.get_json()['format'] # 스프링 서버에서 전달 받은 파일 형식(ex) mp3, mp4, wav)

    voice_file = base64.b64decode(voice)
    input_audio = "./pypost/trans_voice/spring/received_file." + file_format
    with open(input_audio, 'wb') as file:
        file.write(voice_file)

    file_index = f"./Post_AI/pypost/trans_voice/logs/{model_name}/added_IVF1203_Flat_nprobe_1_{model_name}_v2.index"

    convert_path = AI_Convert.Voice_Convert(sid, vc_transform, input_audio, file_index, index_rate)

    with open(convert_path, 'rb') as file:
        wav_data = file.read()
        encoded_voice = base64.b64encode(wav_data).decode()

    return encoded_voice