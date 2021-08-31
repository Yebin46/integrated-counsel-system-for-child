# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import transformers
import pickle
from model import * 
from utils import * 
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler  
import re


class Watcher:
    RASP_DIR = r'\\raspberrypi\pi\ai-makers-kit'
    WORK_DIR = r'C:\Users\Miso CHOI'
    DIRECTORY_WATCH = r'\\raspberrypi\pi\ai-makers-kit\stt_result' #폴더 경로 지정

    def __init__(self):
        self.observer = Observer()  #실행시 Observer객체 생성
  
    def run(self):
        os.chdir(self.RASP_DIR)
        event_handler = Handler() #이벤트 핸들러 객체 생성
        print("Hi handler!")
        self.observer.schedule(event_handler, self.DIRECTORY_WATCH, recursive=False)
        # self.observer.schedule(event_handler, r'.\text_test', recursive=True) #for local laptop 
        self.observer.start()
        # now_path = os.system('pwd')
        # print("No in : ", now_path)
        print("Start observer!")
        try:
            while True:
                    time.sleep(1)
                    print("Wait for file change!")
                        
        except:
            self.observer.stop()
            print("Error")
            self.observer.join()  #스레드가 정상적으로 종료될때까지 기다린다.


class Handler(FileSystemEventHandler):    #이벤트 정의
    @staticmethod
    def on_any_event(event):  #모든 이벤트 발생시
        #!python synthesizer_soomin.py --load_path logdir-tacotron2/kss_preprocess_result+preprocess_result_2021-05-12_20-03-23 --num_speakers 2 --speaker_id 1
        #Observer().stop()
        rasp_dir_w = r'\\raspberrypi\pi\ai-makers-kit\stt_result\stt_result.txt'
        work_dir = r'C:\Users\Miso CHOI'
        print("Context changed!")
        #file_path = glob.glob('../wav_file/*.wav')
        
        time.sleep(1)
        f = open(rasp_dir_w, 'r', encoding='UTF8')
        # f = open(r'.\text_test\content.txt', 'r')
        text = f.read()
        f.close()

        predict(text)
        



def evaluate(sentence):
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size+1]
    VOCAB_SIZE = tokenizer.vocab_size+2
    MAX_LENGTH = 120

    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    # 디코더의 예측 시작
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # 마지막 시점의 예측 단어를 출력에 연결한다
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)

def predict(sentence):
    RASP_DIR = r'\\raspberrypi\pi\ai-makers-kit'
    WORK_DIR = r'C:\Users\Miso CHOI'

    prediction = evaluate(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])
  
    splited = re.split(r'[.?!]', predicted_sentence)
    temp = []
    for i in range(len(predicted_sentence)):
        if predicted_sentence[i] == '.' or predicted_sentence[i] == '?' or predicted_sentence[i] == '!':
            temp.append(predicted_sentence[i])
            
    # splited = predicted_sentence.split('.')
    if len(splited) > 2 :
        predicted_sentence = splited[0]+ temp[0] + splited[-2] + '.'

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))
    
    os.chdir(RASP_DIR)
    fw = open("transformer_result/answer.txt","w")
    fw.write(predicted_sentence)
    fw.close()

    os.chdir(WORK_DIR)

    return predicted_sentence


if __name__ == "__main__":
    with open('tokenizer/tokenizer_total.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size+1]
    VOCAB_SIZE = tokenizer.vocab_size+2
    MAX_LENGTH = 120

    checkpoint_path = "checkpoint/cp-0150.ckpt" 

    tf.keras.backend.clear_session()

    D_MODEL = 256
    NUM_LAYERS = 2
    NUM_HEADS = 8
    DFF = 512
    DROPOUT = 0.1
    
    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        dff=DFF,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

        # 이전에 저장한 가중치를 로드합니다
    model.load_weights(checkpoint_path)

    w = Watcher()
    w.run()



    