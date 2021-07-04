# file name : index.py
# pwd : /project_name/app/main/index.py
 
from flask import Blueprint, request, render_template, flash, redirect, url_for, session
from flask import current_app as app
from flask.globals import g

from app.models import User

# 추가할 모듈이 있다면 추가
main = Blueprint('main', __name__, url_prefix='/')

@main.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('userId'):
        return render_template('/main/index.html')
    else:
        return redirect(url_for('main.login'))

@main.route('/login/', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        userId = request.form['userId']
        password = request.form['password']
        user = User.query.filter_by(userId=userId).first()
        if not user:
            flash('사용자 정보를 찾을 수 없습니다. 다시 입력해주세요.')
        else:
            session.clear()
            session['userId'] = user.id
            return redirect(url_for('main.index'))
    return render_template('/main/index.html')


@main.before_app_request
def load_user():
    userId = session.get('userId')
    if userId is None:
        g.user = None
    else:
        g.user = User.query.get(userId)


@main.route('/logout/')
def logout():
    session.clear()
    return redirect(url_for('main.index'))