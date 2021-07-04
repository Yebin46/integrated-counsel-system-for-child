from flask import Blueprint, request, render_template, flash, redirect, url_for, session
from flask.globals import g
from app.models import User

import glob
import numpy as np
import pandas as pd
import time
from pydub import AudioSegment
from app.analysis.model.dataset import LABEL_DICT
# from app.analysis.model.multi_sentiment_ai import argu
from app.analysis.model import multi_sentiment_ai

analysis = Blueprint('analysis', __name__, url_prefix='/analysis')

@analysis.route('/', methods=["GET"])
def index():
      createPkl()
      #g.emotionAvg = multi_sentiment_ai.argu()
     
      # 최빈값으로 구할 경우
      #g.emotion = " ".join([emotion for emotion, num in LABEL_DICT.items() if mode(g.emotionList) == num])
      # max 값으로 구할 경우
      g.emotion = [emotion for emotion, num in LABEL_DICT.items() if np.where(g.emotionAvg == max(g.emotionAvg))[0] == num]
      #fear, surprise, angry, sad, neutral, happy, disgust
      emotionList = [round(percent, 2) for percent in g.emotionAvg]
      return render_template('/analysis/index.html', emotionList=emotionList)

@analysis.before_app_request
def load_user():
      userId = session.get('userId')
      g.user = User.query.get(userId)
      
def createPkl():
      input_df = pd.DataFrame(columns=['fileName', 'audio'])
      
      # bin 파일 wav 파일로 변환 후 데이터 넣기
      pcmList = glob.glob(r'Z:/ai-makers-kit/pcm_result/*.bin')

      for pcmFile in pcmList[1:len(pcmList)]:
            fileName = pcmFile.split('\\')[1].split('.')[0]
            with open(pcmFile, "rb") as f:
                  data = f.read()
            AudioSegment(data, sample_width=2, frame_rate=16000, channels=1).export("./app/analysis/wav/"+fileName+".wav", format="wav")
            audio = AudioSegment.from_wav("./app/analysis/wav/"+fileName+".wav")
            audio = audio.set_channels(1)
            audio = np.array(audio.get_array_of_samples())
            cur_row = [fileName, audio]
            input_df.loc[len(input_df)] = cur_row

      # 텍스트 데이터 넣기
      text = []

      f = open(r'z:/ai-makers-kit/pcm_result/pcm_result.txt', 'r', encoding = 'utf8')
      lines = f.readlines()
      for line in lines[1:len(lines)]:
            line = line.strip('\n')
            text.append(line)
      f.close()

      input_df['sentence'] = text

      # null 값 제거
      input_df = input_df[input_df.sentence != '']

      print(input_df)

      input_df.to_pickle('./app/analysis/predict.pkl')

def mode(list):
    count = 0
    mode = 0
    for x in list: 
        if list.count(x) > count:
            count = list.count(x)
            mode = x
    return mode