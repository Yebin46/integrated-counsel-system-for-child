from __future__ import print_function
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import MicrophoneStream as MS
import ex1_kwstest as kws
import ex4_getText2VoiceStream as tts
# import ex6_queryVoice as dss
import getVoice2Text_yb as gv2t
import time
import os
import glob

class Watcher:
     DIRECTORY_WATCH = '../tts_result' # 관찰할 폴더 지정
  
     def __init__(self):
          self.observer = Observer()  # 실행시 Observer객체 생성
  
     def run(self):
          event_handler = Handler() # 이벤트 핸들러 객체 생성
          self.observer.schedule(event_handler, self.DIRECTORY_WATCH, recursive=False) 
          self.observer.start()
          print("Start TTS_result check..\n")
          try:
                    while True:
                          time.sleep(1)
                          print("Wait for response..\n")
          except:
                    self.observer.stop()
                    print("Error")
                    self.observer.join()  #스레드가 정상적으로 종료될때까지 기다린다.
'''
class Handler(FileSystemEventHandler):    #이벤트 정의
          @staticmethod
          def on_any_event(event):  #모든 이벤트 발생시
            #!python synthesizer_soomin.py --load_path logdir-tacotron2/kss_preprocess_result+preprocess_result_2021-05-12_20-03-23 --num_speakers 2 --speaker_id 1
            print("IU RESPONSED!\n")
            #file_path = glob.glob('../wav_file/*.wav')
            file_path = '../tts_result/answer.wav'
            time.sleep(4)
            #print(type(file_path))
            MS.play_file(file_path)
'''

class MyEventHandler(FileSystemEventHandler):
          def __init__(self, observer):
              self.observer = observer
              
          def on_any_event(self, event):
              self.observer.stop()
              print("IU RESPONSED!\n")
              file_path = '../tts_result/answer.wav'
              time.sleep(6)
              MS.play_file(file_path)
              time.sleep(3)
def save_text(resultText):
    f = open("../stt_result/stt_result.txt", 'w') #여기 폴더 이름 txt file의 위치로 변경하기
    f.write(resultText)
    f.close()
    p = open("../pcm_result/pcm_result.txt", 'a')
    p.write(resultText+'\n')
    p.close()
    
def main():
	#Example8 KWS+STT+DSS

	KWSID = ['기가지니', '지니야', '친구야', '자기야']
	end_word = '끝낼게'
	recog=kws.test(KWSID[2]) # 친구야 인식
	final_mesg = "../python3/ex_wav_data/thank_you_for_sharing_your_story.wav"
	final_mesg2 = "../python3/ex_wav_data/if_you_have_gomin_come_next.wav"
	count = 0
	
	#final_mesg3 = "../python3/ex_wav_data/gomin_haegurl_nonstop.wav"
	while(1):
		if recog == 200:
			time.sleep(1)
			print('Start STT...\n')
			count += 1
			text = gv2t.getVoice2Text(count)
			
			#if text == '':
			#	print('앗 잘 못들었다ㅠㅠ 다시 한 번 얘기해줘라!\n\n\n')			
			#	#time.sleep(2)
			#	count += 1
			#	text = gv2t.getVoice2Text(count)
			
			while text == '':
                            print('앗 잘 못들었다ㅠㅠ 다시 한 번 얘기해줘라!\n\n\n')
                            text = gv2t.getVoice2Text(count)
                            #print('앗 잘 못들었다ㅠㅠ 다시 한 번 얘기해줘라!\n\n\n')
                            
			save_text(text)

			if end_word in text:
                            MS.play_file(final_mesg) #better!
                            MS.play_file(final_mesg2)
                            #MS.play_file(final_mesg3)
                            break
			
			observer = Observer()
			event_handler = MyEventHandler(observer)
			observer.schedule(event_handler, r'../tts_result', recursive=False)
			observer.start()
			observer.join()
			#w = Watcher()
			#w.run()
			#if text == '':
			#	print('질의한 내용이 없습니다.\n\n\n')			
				#time.sleep(2)
				

if __name__ == '__main__':
    #[os.remove(f) for f in glob.glob("../pcm_result/*.bin")]
    #os.remove("../pcm_result/pcm_result.txt")
    main()
