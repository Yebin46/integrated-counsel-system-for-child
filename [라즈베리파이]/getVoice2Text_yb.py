#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""내담자의 발화 STT해서 공유 폴더의 input_content.txt에 저장 """
from __future__ import print_function
import grpc
import gigagenieRPC_pb2
import gigagenieRPC_pb2_grpc
import MicrophoneStream as MS
import user_auth as UA
import audioop
import os
from ctypes import *

HOST = 'gate.gigagenie.ai'
PORT = 4080
RATE = 16000
CHUNK = 512

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
  dummy_var = 0
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = cdll.LoadLibrary('libasound.so')
asound.snd_lib_error_set_handler(c_error_handler)

def generate_request(count): #마이크에서 가져온 데이터를 AI KT 에듀팩 STT API에 입력할 수 있도록 변환해주는 함수
    with MS.MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        
        for content in audio_generator:
            message = gigagenieRPC_pb2.reqVoice()
            message.audioContent = content
            binPath = '/home/pi/ai-makers-kit/pcm_result/pcm-'+str(count)+'.bin'
            with open(binPath,"ab") as f:
                  f.write(message.audioContent)
            yield message

            rms = audioop.rms(content,2)
            #print_rms(rms)

def getVoice2Text(count): #변환된 데이터 값을 가져와서 STT API에 입력하여 텍스트를 출력
    print ("\n\n음성인식을 시작합니다.\n\n종료하시려면 Ctrl+\ 키를 누루세요.\n\n\n")
    channel = grpc.secure_channel('{}:{}'.format(HOST, PORT), UA.getCredentials())
    stub = gigagenieRPC_pb2_grpc.GigagenieStub(channel)
    request = generate_request(count)
    resultText = ''
    for response in stub.getVoice2Text(request):
        if response.resultCd == 200: # partial
            print('resultCd=%d | recognizedText= %s'
                % (response.resultCd, response.recognizedText))
            resultText = response.recognizedText
        elif response.resultCd == 201: # final
            print('resultCd=%d | recognizedText= %s'
                % (response.resultCd, response.recognizedText))
            resultText = response.recognizedText
            break
        else:
            print('resultCd=%d | recognizedText= %s'
                % (response.resultCd, response.recognizedText))
            break
    print ("\n\n인식결과: %s \n\n\n" % (resultText))
    
    '''
    f = open("../stt_result/stt_result.txt", 'w') #여기 폴더 이름 txt file의 위치로 변경하기
    f.write(resultText)
    f.close()
    p = open("../pcm_result/pcm_result.txt", 'a')
    p.write(resultText+'\n')
    p.close()
    '''
    return resultText

def main():
  # STT
  text = getVoice2Text(count)
'''
if __name__ == '__main__':
  main()
'''