from app import db

class User(db.Model):
    __tablename__ = 'user'
    # id는 자동으로 증가하는 User 모델의 기본키
    id = db.Column(db.Integer, primary_key=True)
    userId = db.Column(db.String(150), unique = True, nullable = False)
    password = db.Column(db.String(200), nullable = False)
    userName = db.Column(db.String(120), nullable = False)