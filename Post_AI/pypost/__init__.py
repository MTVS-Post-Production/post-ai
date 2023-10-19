from flask import Flask

def create_app():
    app = Flask(__name__)

    from pypost.views import main_views, voice_views, pose_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(pose_views.bp)
    app.register_blueprint(voice_views.bp)

    return app