from flask import Flask
from .views import main_views, voice_views, pose_views

def create_app():
    app = Flask(__name__)

    app.register_blueprint(main_views.bp)
    app.register_blueprint(pose_views.bp)
    app.register_blueprint(voice_views.bp)

    return app