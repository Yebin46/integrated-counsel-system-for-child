# integrated-counsel-system-for-child
자연어처리, 음성합성, 감정분석을 활용한 아동청소년 심리상담 통합시스템 개발 프로젝트

## Project Diagram
전체적인 프로그램 모식도

## Natural Language Processing
자연어처리를 활용한 상담 답변 텍스트(.txt) 생성 (transformer)

## Voice Synthesis
상담 답변 텍스트(.txt)를 음성(.wav)으로 합성 (multi-speaker-tacotron2)

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