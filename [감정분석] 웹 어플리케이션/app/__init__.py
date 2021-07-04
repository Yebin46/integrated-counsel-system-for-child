from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

import config
import os
import glob

db = SQLAlchemy()
migrate = Migrate()

def create_app():

    app = Flask(__name__)

    # config 파일에 작성한 항목을 app.config 환경변수로 불러오기
    app.config.from_object(config)

    # wav 디렉토리 초기화
    # [os.remove(f) for f in glob.glob("./app/analysis/wav/*.wav")]

    # 초기화
    db.init_app(app)
    migrate.init_app(app, db)
    from . import models

    from .main import index
    app.register_blueprint(index.main)

    from .analysis.index import analysis as analysis
    app.register_blueprint(analysis)

    return app