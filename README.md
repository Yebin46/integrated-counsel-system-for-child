# integrated-counsel-system-for-child
자연어처리, 음성합성, 감정분석을 활용한 아동청소년 심리상담 통합시스템 개발 프로젝트

## Project Diagram

![project diagram](https://user-images.githubusercontent.com/58056141/124941849-15065080-e046-11eb-88a2-ac56a929c277.JPG)

## Natural Language Processing
자연어처리를 활용한 상담 답변 텍스트(.txt) 생성 (transformer)

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
> > Flask
> > SQLite
> > html/css
> > pytorch
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
