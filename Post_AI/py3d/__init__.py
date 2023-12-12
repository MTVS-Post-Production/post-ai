import os, sys

flask_dir = os.getcwd() + '\\py3d'
sys.path.append(flask_dir)

from flask import Flask, request, jsonify
from one_click_process import Make_3d_Modeling
import base64
from google.cloud import storage
from glob import glob
import torch
import time
import shutil

# Google Storage 등록
KEY_PATH = "/home/meta-ai2/Magic123/py3d/google-storage.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= KEY_PATH

app = Flask(__name__)

@app.route('/')
def index():
    return "3d 모델링 제작 서버입니다."

@app.route('/objmaker', methods=['GET', 'POST'])
def main():
    os.makedirs('./py3d/spring', exist_ok=True)

    image = request.get_json()['image']
    image_name = request.get_json()['image_name']
    user_id = request.get_json()['user_id']
    
    print(f'이미지 이름: {image_name}')
    bucket_name = 'image_production'

    try:
        image_file = base64.b64decode(image)
        input_image = f"./py3d/spring/{image_name}.jpg"

        with open(input_image, 'wb') as file:
            file.write(image_file)

    except:
        print("이미지 코드 에러")
        pass

    time.sleep(0.5)

    destination_albedo = f'{user_id}/{image_name}/albedo.png'
    destination_mesh_mtl = f'{user_id}/{image_name}/mesh.mtl'
    destination_mesh_obj = f'{user_id}/{image_name}/mesh.obj'
    destination_mp4 = f'{user_id}/{image_name}/mesh_example.mp4'

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_albedo = bucket.blob(destination_albedo)
    blob_mesh_mtl = bucket.blob(destination_mesh_mtl)
    blob_mesh_obj = bucket.blob(destination_mesh_obj)
    blob_mp4 = bucket.blob(destination_mp4)

    # 중복 방지를 위한 파일 정리
    try:
        blob_albedo.reload()
        blob_mesh_mtl.reload()
        blob_mesh_obj.reload()

        generation_match_precondition_albedo = blob_albedo.generation
        blob_albedo.delete(if_generation_match=generation_match_precondition_albedo)
        print(f"Blob {destination_albedo} deleted.")

        time.sleep(0.5)

        generation_match_precondition_mtl = blob_mesh_mtl.generation
        blob_mesh_mtl.delete(if_generation_match=generation_match_precondition_mtl)
        print(f"Blob {destination_mesh_mtl} deleted.")

        time.sleep(0.5)

        generation_match_precondition_obj = blob_mesh_obj.generation
        blob_mesh_obj.delete(if_generation_match=generation_match_precondition_obj)
        print(f"Blob {destination_mesh_obj} deleted.")

        time.sleep(0.5)

        generation_match_precondition_mp4 = blob_mp4.generation
        blob_mp4.delete(if_generation_match=generation_match_precondition_mp4)
        print(f"Blob {destination_mp4} deleted.")

        time.sleep(0.5)
    
    except:
        print(f"Blob {destination_albedo} / {destination_mesh_mtl} / {destination_mesh_obj} / {destination_mp4} does not exists.")
        pass

    torch.cuda.empty_cache()
    obj = Make_3d_Modeling(filepath=input_image, is_bool=False, token="")
    obj.run()

    time.sleep(0.5)

    # 생성된 파일 옮기기
    os.makedirs(f'./run_results/{image_name}', exist_ok=True)

    shutil.copyfile(f'./run/{image_name}/outputs/retouch/mesh/albedo.png', f'./run_results/{image_name}/albedo.png')
    shutil.copyfile(f'./run/{image_name}/outputs/retouch/mesh/mesh.mtl', f'./run_results/{image_name}/mesh.mtl')
    shutil.copyfile(f'./run/{image_name}/outputs/retouch/mesh/mesh.obj', f'./run_results/{image_name}/mesh.obj')
    shutil.copyfile(f'./run/{image_name}/outputs/retouch/results/retouch_ep0050_lambertian.mp4', f'./run_results/{image_name}/example.mp4')

    results_path = f'./run_results/{image_name}'

    blob_albedo.upload_from_filename(results_path + '/albedo.png')

    time.sleep(0.5)
    blob_mesh_mtl.upload_from_filename(results_path + '/mesh.mtl')

    time.sleep(0.5)
    blob_mesh_obj.upload_from_filename(results_path + '/mesh.obj')

    time.sleep(0.5)
    blob_mp4.upload_from_filename(results_path + '/example.mp4')
    
    url_albedo = f"https://storage.googleapis.com/image_production/{user_id}/{image_name}/albedo.png"
    url_mesh_mtl = f"https://storage.googleapis.com/image_production/{user_id}/{image_name}/mesh.mtl"
    url_mesh_obj = f"https://storage.googleapis.com/image_production/{user_id}/{image_name}/mesh.obj"
    url_mesh_example = f"https://storage.googleapis.com/image_production/{user_id}/{image_name}/example.mp4"

    urls = {'albedo':url_albedo, 'mesh_mtl':url_mesh_mtl, 'mesh_obj':url_mesh_obj, 'url_mesh_example':url_mesh_example}
    print(urls)

    return jsonify(urls)


@app.route('/urlmaker', methods=['GET', 'POST'])
def main2():
    image = request.get_json()['image']
    image_name = request.get_json()['image_name']
    user_id = request.get_json()['user_id']
    
    print(f'이미지 이름: {image_name}')

    try:
        image_file = base64.b64decode(image)
        input_image = f"./py3d/spring/{image_name}.jpg"

        with open(input_image, 'wb') as file:
            file.write(image_file)

    except:
        print("이미지 코드 에러")
        pass

    time.sleep(10)

    url_albedo = f"https://storage.googleapis.com/image_production/{user_id}/{image_name}/albedo.png"
    url_mesh_mtl = f"https://storage.googleapis.com/image_production/{user_id}/{image_name}/mesh.mtl"
    url_mesh_obj = f"https://storage.googleapis.com/image_production/{user_id}/{image_name}/mesh.obj"
    url_mesh_example = f"https://storage.googleapis.com/image_production/{user_id}/{image_name}/example.mp4"
    
    urls = {'albedo':url_albedo, 'mesh_mtl':url_mesh_mtl, 'mesh_obj':url_mesh_obj, 'url_mesh_example':url_mesh_example}
    print(urls)

    return jsonify(urls)