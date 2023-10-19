from flask import Blueprint

bp = Blueprint('main', __name__, url_prefix='/')

# mp4 파일을 받아 pose estimation을 진행
@bp.route('/')
def index_main():
    return "AI 메인 화면입니다."