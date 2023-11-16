import os, sys

flask_dir = os.getcwd() + "\\pypost"
sys.path.append(flask_dir)

from flask import Blueprint, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import base64

# 부호를 계산하기
def calculate_sign(a, b, c):
    sign_x1 = c[0] - b[0]
    sign_x2 = a[0] - b[0]

    # (x1, y1), (x2, y2), (x3, y3)일때, x3-x2와 x1-x2의 부호가 같으면 1을 리턴, 아니면 0을 리턴
    if sign_x1 >= 0 and sign_x2 >= 0:
        return 1
    elif sign_x1 >= 0 and sign_x2 < 0:
        return 0
    elif sign_x1 < 0 and sign_x2 >= 0:
        return 0
    else:
        return 1

# image로 각도 구하기
def three_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # return angle, 만약 부호가 다르면 각도를 반대로 재야 함
    if calculate_sign(a, b, c) == 1:
        return angle
    else:
        rev_angle = 360 - angle
        return rev_angle

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

df = pd.read_csv('rev_pose_data.csv')
df = df[df['label'] != 5]
cos_df = df.iloc[:, :-1]

pose_dict = {0:'catch', 1:'jump', 2:'lift', 3:'run', 4:'sit', 5:'spell', 6:'walk'}

degrees = {'left_elbow': [], 'right_elbow': [], 'left_armpit': [], 'right_armpit': [], 'left_hip_outside': [],
           'right_hip_outside': [], 'left_hip_inside': [], 'right_hip_inside': [], 'left_knee': [], 'right_knee': []}

bp = Blueprint('checkpose', __name__, url_prefix='/checkpose')

# mp4 파일을 받아 pose estimation을 진행
@bp.route('/', methods=['GET', 'POST'])
def video_pose():
    pose = request.get_json()['pose']  # 스프링 서버에서 전달 받은 base64로 인코딩된 파일
    pose_video = base64.b64decode(pose)

    # 디코딩한 mp4 파일을 폴더에 저장
    os.makedirs('./pypost/pose_estimation', exist_ok=True)
    with open('./pypost/pose_estimation/received_file.mp4', 'wb') as file:
        file.write(pose_video)

    cap = cv2.VideoCapture('./pypost/pose_estimation/received_file.mp4')
    pose_list = []

    # 영상을 기반으로 추론
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            image = cv2.resize(image, (720,720))
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            height, width, _ = image.shape

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                try:
                    landmarks = results.pose_landmarks.landmark

                    # 좌표값 구하기
                    def get_corrs(pose_num):
                        corr = [landmarks[pose_num].x * width, landmarks[pose_num].y * height, landmarks[pose_num].z * width]
                        return corr

                    # 좌표 구하기
                    left_shoulder = get_corrs(11)  # 왼쪽 어깨 (11)
                    right_shoulder = get_corrs(12)  # 오른쪽 어깨 (12)

                    left_elbow = get_corrs(13)  # 왼쪽 팔꿈치 (13)
                    right_elbow = get_corrs(14)  # 오른쪽 팔꿈치 (14)

                    left_wrist = get_corrs(15)  # 왼쪽 손목 (15)
                    right_wrist = get_corrs(16)  # 오른쪽 손목 (16)

                    left_hip = get_corrs(23)  # 왼쪽 골반 (23)
                    right_hip = get_corrs(24)  # 오른쪽 골반 (24)

                    left_knee = get_corrs(25)  # 왼쪽 무릎 (25)
                    right_knee = get_corrs(26)  # 오른쪽 무릎 (26)

                    left_ankle = get_corrs(27)  # 왼쪽 발목 (27)
                    right_ankle = get_corrs(28)  # 오른쪽 발목 (28)

                    # 각도 구하기
                    left_elbow_degree = three_angle(left_shoulder, left_elbow, left_wrist)  # 11 - 13 - 15
                    right_elbow_degree = three_angle(right_shoulder, right_elbow, right_wrist)  # 12 - 14 - 16

                    left_shoulder_degree = three_angle(left_elbow, left_shoulder, left_hip)  # 13 - 11 - 23
                    right_shoulder_degree = three_angle(right_elbow, right_shoulder, right_hip)  # 14 - 12 - 24

                    left_hip_degree_out = three_angle(left_shoulder, left_hip, left_knee)  # 11 - 23 - 25
                    right_hip_degree_out = three_angle(right_shoulder, right_hip, right_knee)  # 12 - 24 - 26

                    left_hip_degree_in = three_angle(right_hip, left_hip, left_knee)  # 24 - 23 - 25
                    right_hip_degree_in = three_angle(left_hip, right_hip, right_knee)  # 23 - 24 - 26

                    left_knee_degree = three_angle(left_hip, left_knee, left_ankle)  # 23 - 25 - 27
                    right_knee_degree = three_angle(right_hip, right_knee, right_ankle)  # 24 - 26 - 28

                    # 딕셔너리에 저장, 이후 dataframe로 치환하기 위함
                    degrees['left_elbow'].extend([left_elbow_degree])
                    degrees['left_armpit'].extend([left_shoulder_degree])
                    degrees['left_hip_outside'].extend([left_hip_degree_out])
                    degrees['left_hip_inside'].extend([left_hip_degree_in])
                    degrees['left_knee'].extend([left_knee_degree])

                    degrees['right_elbow'].extend([right_elbow_degree])
                    degrees['right_armpit'].extend([right_shoulder_degree])
                    degrees['right_hip_outside'].extend([right_hip_degree_out])
                    degrees['right_hip_inside'].extend([right_hip_degree_in])
                    degrees['right_knee'].extend([right_knee_degree])

                    test_df = pd.DataFrame(degrees)

                    pose_data = np.array(cos_df)
                    test_data = np.array(test_df)

                    similarity = cosine_similarity(pose_data, test_data)
                    predicted_labels = np.argmax(similarity, axis=0)

                    result = Counter(predicted_labels)
                    max_result, _ = result.most_common(1)[0]

                    text = pose_dict[int(df.iloc[max_result]['label'])]

                    # print(f"행동:{text}, 행번호:{max_result}")
                    pose_list.append(text)

                except Exception as e:
                    print(e)
                    pass

    test_df = pd.DataFrame(degrees)
    print("총 frame 수:", test_df.index[-1])

    # 영상을 전부 기록 후 5프레임마다 측정
    for i in range(test_df.index[-1] // 5):
        pose_data = np.array(cos_df)

        # 코사인 유사도
        similarity_lst = []
        for k in range(i*5, (i+1)*5):
            test_data = np.array(test_df.iloc[k:(k+1), :])
            similarity = cosine_similarity(pose_data, test_data)
            print(similarity)
            predicted_labels = tuple(np.argmax(similarity, axis=0))
            print(predicted_labels)
            similarity_lst.append(predicted_labels)

        result = Counter(similarity_lst)
        print(result)

        # 경우의 수: [5], [4, 1], [3, 1, 1], [2, 2, 1], [2, 1, 1, 1], [1, 1, 1, 1, 1]
        if len(set(result.values())) % 2 != 0:
            if 3 not in set(result.values()):
                # 만약 동률인 경우가 있으면, 가장 추정 확률이 높았던 동작을 리턴
                max_result = max(result.items(), key=lambda x: (x[1], max(x[0])))
            else:
                max_result, _ = result.most_common(1)[0]
        else:
            max_result, _ = result.most_common(1)[0]  # (가장 많이 나온 행, 나온 횟수)

        text = pose_dict[int(df.iloc[max_result[0]]['label'])]

        print(f"행동:{text}, 번호:{max_result[0]}")
        pose_list.append(text)


    print(pose_list)
    cap.release()
    cv2.destroyAllWindows()

    # response_data = list(set(pose_list))
    response_data = pose_list

    print("응답 전송:", response_data)

    return jsonify(response_data)



# 문자열을 받아 pose 추천을 진행
@bp.route('/string', methods=['GET', 'POST'])
def string_pose():
    pose = request.get_json()['pose']
    
    return jsonify([pose])