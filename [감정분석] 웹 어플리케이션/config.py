'''
config 파일추가: db.sqlite 데이터베이스 파일 root 디렉토리에 저장
'''
import os

base_dir = os.path.dirname(__file__)

# 사용할 DB URI
SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(base_dir, 'db.sqlite'))
# 비지니스 로직이 끝날때 Commit 실행(DB반영)
# SQLALCHEMY_COMMIT_ON_TEARDOWN = True
# 수정사항에 대한 TRACK(이벤트 처리 옵션)
SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = 'jqiowejrojzxcovnklqnweiorjqwoijroi'