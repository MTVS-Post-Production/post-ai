# 주요 기능
### Animation Search from Pose Classification
- 통신을 통해 동작을 전달받으면 로컬영역에 저장 후 추론 수행
- 만약 길이가 20초보다 길면 20초 단위로 영상 분할(겹치는 영역은 10초) 후 추론

### Voice Change
- LiveLink를 통해 음성 업로드시 선택한 모델에 따라서 검색 기반 음성 변조 수행

### Image to 3d Object
- 2d image 업로드 시 해당 image를 기반으로 3d Object Mesh를 생성
- 약 1시간 소요

# 참고 자료
### 오픈소스(github)
- https://github.com/google/mediapipe
- https://huggingface.co/docs/transformers/v4.32.0/ko/tasks/video_classification
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- https://github.com/guochengqian/Magic123
