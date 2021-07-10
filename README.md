# integrated-counsel-system-for-child
자연어처리, 음성합성, 감정분석을 활용한 아동청소년 심리상담 통합시스템 개발 프로젝트

## Project Diagram

![project diagram](https://user-images.githubusercontent.com/58056141/124941849-15065080-e046-11eb-88a2-ac56a929c277.JPG)

## Hardware
- 스피커의 하드웨어는 라즈베리파이 3B+ 모델을 사용하여 제작
- STT를 위한 음성인식모듈을 사용하기 위해 마이크 보드 및 스피커를 보이스 쉴드에 연결

## Process
* 본 프로젝트에서는 자연어처리, 음성합성, 감정분석 총 3가지 모델 사용. 와이파이를 통해 각 모델을 실행하는 로컬PC에서 samba 서버를 사용하여 라즈베리파이의 공유 폴더에 접근함
* 개발자가 다른 조치를 취하지 않아도 모든 프로세스가 연속적으로 이뤄지도록 하기 위해 Watchdog 패키지를 이용하여 input파일이 새로 생성되는 이벤트가 발생하면 자동으로 모델의 input으로 들어가도록 개발함

## Natural Language Processing
자연어처리를 활용한 상담 답변 텍스트(.txt) 생성 (transformer)
#### Dataset
심리상담 질문과 정신건강의학과 전문의 외 5명의 전문가 답변, 고민 Q&A 파트 질문과 여성가족부의 답변을 크롤링하여 사용
* 네이버 지식 in 데이터셋은 저작권 때문에 공개하지 않습니다.
* '웰니스 대화 스크립트 데이터셋'은 아래 링크(AIhub)에서 받으실 수 있습니다. 
   (https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-006)
* 'Chatbot_data'는 아래 링크에서 받으실 수 있습니다.
   (https://github.com/songys/Chatbot_data)
   

#### Transformer for local PC
* 사용자의 음성이 라즈베리파이에서 STT된 결과(stt_result.txt)가 공유 폴더 내에 저장되면 자연어처리 모델의 Watchdog observer가 이벤트를 인식하여 Transformer의 input으로 들어가도록 구현
* Transformer의 결과로 반환된 답변(answer.txt)을 공유폴더에 생성
> 진행 방법
> > 1. local PC에서 가상환경을 만들어 requirements.txt 설치
> > 2. test.py에서 사용자에 맞게 경로 변경 : 
         RASP_DIR, WORK_DIR, DIRECTORY_WATCH(stt_result.txt 생성 경로) 
> > 3. 터미널에서 아래 코드 실행

'''
python test.py
'''




## Voice Synthesis
상담 답변 텍스트(.txt)를 음성(.wav)으로 합성 (multi-speaker-tacotron2)
* iu 데이터셋은 저작권 때문에 공개하지 않습니다.
* KSS 데이터셋은 아래 링크에서 받으실 수 있습니다.
    * https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset
> Tacotron for local PC code
> > Google Colab에서 Tacotron2를 training한 후 checkpoint를 저장하여 local PC에서 사용.
> > Watchdog를 활용해 Transformer의 output인 answer.txt가 업데이트되면 자동으로 음성 합성 모델의 input으로 들어가도록 구현 (synthesizer.py 참조)
> > > 진행 방법
> > > 1. local PC에서 가상환경을 만들어 requirement.txt 설치
> > > 2. checkpoint를 알맞은 PATH에 저장한 후 터미널에서 아래 코드 실행
'''
python synthesizer.py --load_path CHECKPOINT_PATH
'''

## Sentiment Analysis
내담자의 음성과 텍스트를 사용해 감정을 분석 (multi-modal-transformer)

### Sentiment Analysis Website
웹 어플리케이션 형태로 내담자의 감정 분석 결과 레포트 제공
#### Project stack
* Flask
* SQLite
* html/css/js
* pytorch
#### Component Diagram

![component diagram](https://user-images.githubusercontent.com/58056141/124938738-8264b200-e043-11eb-891b-853ea74477d7.JPG)

#### Gallery
* 초기 화면
   ![main](https://user-images.githubusercontent.com/58056141/124940758-3581db00-e045-11eb-9035-a5ed0c5c13c3.JPG)
* 개인 프로필
   ![profile](https://user-images.githubusercontent.com/58056141/124940795-3e72ac80-e045-11eb-91a6-8d1e70abfdd6.JPG)
* 감정 분석 레포트
   ![report](https://user-images.githubusercontent.com/58056141/124940831-46cae780-e045-11eb-9ead-2ce4c54595c9.JPG)

### Based on
* Transformer
    * 
* Multi-Speaker-Tacotron2
    * https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS
    * https://github.com/khw11044/Tacotron-in-colab
* Multi-Modal-Transformer
    * https://github.com/youngbin-ro/audiotext-transformer
    * https://github.com/Donghwa-KIM/audiotext-transformer
