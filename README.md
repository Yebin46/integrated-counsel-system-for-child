# integrated-counsel-system-for-child
자연어처리, 음성합성, 감정분석을 활용한 아동청소년 심리상담 통합시스템 개발 프로젝트

## Project Diagram
전체적인 프로그램 모식도

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

### Based on
* Transformer
    * 
* Multi-Speaker-Tacotron2
    * https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS
    * https://github.com/khw11044/Tacotron-in-colab
* Multi-Modal-Transformer
    * 