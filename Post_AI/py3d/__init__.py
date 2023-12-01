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

# Google Storage 등록
KEY_PATH = "/home/meta-3/Magic123/py3d/google-storage.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= KEY_PATH

app = Flask(__name__)

@app.route('/')
def index():
    return "3d 모델링 제작 서버입니다."

@app.route('/objmaker', methods=['GET', 'POST'])
def main():
    image = request.get_json()['image']
    image_name = request.get_json()['image_name']
    user_id = request.get_json()['user_id']
    
    print(f'이미지 이름: {image_name}')
    print(f'이미지 이름: {user_id}')

    bucket_name = 'image_production'

    image_file = base64.b64decode(image)
    input_image = f"./py3d/spring/{image_name}.jpg"

    with open(input_image, 'wb') as file:
        file.write(image_file)

    time.sleep(0.5)

    destination_albedo = f'{user_id}/{image_name}/albedo.png'
    destination_mesh_mtl = f'{user_id}/{image_name}/mesh.mtl'
    destination_mesh_obj = f'{user_id}/{image_name}/mesh.obj'

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_albedo = bucket.blob(destination_albedo)
    blob_mesh_mtl = bucket.blob(destination_mesh_mtl)
    blob_mesh_obj = bucket.blob(destination_mesh_obj)

    torch.cuda.empty_cache()
    obj = Make_3d_Modeling(filepath=input_image, is_bool=False, token="")
    obj.run()

    time.sleep(0.5)
        
    new_path = f'./run/{image_name}'
    mesh_path = new_path + '/outputs/retouch/mesh'

    blob_albedo.upload_from_filename(mesh_path + '/albedo.png')

    time.sleep(0.5)
    blob_mesh_mtl.upload_from_filename(mesh_path + '/mesh.mtl')

    time.sleep(0.5)
    blob_mesh_obj.upload_from_filename(mesh_path + '/mesh.obj')
    # # blob_mp4.upload_from_filename(results_path)

    url_albedo = f"https://storage.cloud.google.com/image_production/{user_id}/{image_name}/albedo.png"
    url_mesh_mtl = f"https://storage.cloud.google.com/image_production/{user_id}/{image_name}/mesh.mtl"
    url_mesh_obj = f"https://storage.cloud.google.com/image_production/{user_id}/{image_name}/mesh.obj"

    urls = {'albedo':url_albedo, 'mesh_mtl':url_mesh_mtl, 'mesh_obj':url_mesh_obj}
    print(urls)

    return jsonify(urls)