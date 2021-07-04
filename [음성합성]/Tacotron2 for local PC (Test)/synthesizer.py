# coding: utf-8

import io
import os
import re
import librosa
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from functools import partial

from hparams import hparams
from tacotron2 import create_model, get_most_recent_checkpoint
from utils.audio import save_wav, inv_linear_spectrogram, inv_preemphasis, inv_spectrogram_tensorflow
from utils import PARAMS_NAME, load_json, load_hparams, add_prefix, add_postfix, get_time, parallel_run, makedirs, str2bool

from text.korean import tokenize
from text import text_to_sequence, sequence_to_text
from datasets.datafeeder_tacotron2 import _prepare_inputs
import warnings

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler  
import time

warnings.simplefilter(action='ignore', category=FutureWarning)
tf.logging.set_verbosity(tf.logging.ERROR)

class Synthesizer(object):
    def close(self):
        tf.reset_default_graph()
        self.sess.close()

    def load(self, checkpoint_path, num_speakers=2, checkpoint_step=None, inference_prenet_dropout=True,model_name='tacotron'):
        self.num_speakers = num_speakers

        if os.path.isdir(checkpoint_path):
            load_path = checkpoint_path
            checkpoint_path = get_most_recent_checkpoint(checkpoint_path, checkpoint_step)
        else:
            load_path = os.path.dirname(checkpoint_path)

        print('Constructing model: %s' % model_name)

        inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')

        batch_size = tf.shape(inputs)[0]
        speaker_id = tf.placeholder_with_default(
                tf.zeros([batch_size], dtype=tf.int32), [None], 'speaker_id')

        load_hparams(hparams, load_path)
        hparams.inference_prenet_dropout = inference_prenet_dropout
        with tf.variable_scope('model') as scope:
            self.model = create_model(hparams)

            self.model.initialize(inputs=inputs, input_lengths=input_lengths, num_speakers=self.num_speakers, speaker_id=speaker_id,is_training=False)
            self.wav_output = inv_spectrogram_tensorflow(self.model.linear_outputs,hparams)

        print('Loading checkpoint: %s' % checkpoint_path)

        sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=2)
        sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_path)

    def synthesize(self,
            texts=None, tokens=None,
            base_path=None, paths=None, speaker_ids=None,
            start_of_sentence=None, end_of_sentence=True,
            pre_word_num=0, post_word_num=0,
            pre_surplus_idx=0, post_surplus_idx=1,
            use_short_concat=False,
            base_alignment_path=None,
            librosa_trim=False,
            attention_trim=True,
            isKorean=True):
        # Possible inputs:
        # 1) text=text
        # 2) text=texts
        # 3) tokens=tokens, texts=texts # use texts as guide

        if type(texts) == str:
            texts = [texts]

        if texts is not None and tokens is None:
            sequences = np.array([text_to_sequence(text) for text in texts])
            sequences = _prepare_inputs(sequences)
        elif tokens is not None:
            sequences = tokens

        #sequences = np.pad(sequences,[(0,0),(0,5)],'constant',constant_values=(0))  # case by case ---> overfitting?
        
        if paths is None:
            paths = [None] * len(sequences)
        if texts is None:
            texts = [None] * len(sequences)

        time_str = get_time()
        def plot_and_save_parallel(wavs, alignments,mels):

            items = list(enumerate(zip(wavs, alignments, paths, texts, sequences,mels)))

            fn = partial(
                    plot_graph_and_save_audio,
                    base_path=base_path,
                    start_of_sentence=start_of_sentence, end_of_sentence=end_of_sentence,
                    pre_word_num=pre_word_num, post_word_num=post_word_num,
                    pre_surplus_idx=pre_surplus_idx, post_surplus_idx=post_surplus_idx,
                    use_short_concat=use_short_concat,
                    librosa_trim=librosa_trim,
                    attention_trim=attention_trim,
                    time_str=time_str,
                    isKorean=isKorean)
            return parallel_run(fn, items,desc="plot_graph_and_save_audio", parallel=False)

        #input_lengths = np.argmax(np.array(sequences) == 1, 1)+1
        input_lengths = [np.argmax(a==1)+1 for a in sequences]

        fetches = [
                #self.wav_output,
                self.model.linear_outputs,
                self.model.alignments,   #  # batch_size, text length(encoder), target length(decoder)
                self.model.mel_outputs,
        ]

        feed_dict = { self.model.inputs: sequences, self.model.input_lengths: input_lengths, }


        if speaker_ids is not None:
            if type(speaker_ids) == dict:
                speaker_embed_table = sess.run(
                        self.model.speaker_embed_table)

                speaker_embed =  [speaker_ids[speaker_id] * speaker_embed_table[speaker_id] for speaker_id in speaker_ids]
                feed_dict.update({ self.model.speaker_embed_table: np.tile() })
            else:
                feed_dict[self.model.speaker_id] = speaker_ids

        wavs, alignments,mels = self.sess.run(fetches, feed_dict=feed_dict)
        audio_out_result = plot_and_save_parallel(wavs, alignments,mels=mels)

        return audio_out_result

def plot_graph_and_save_audio(args,
        base_path=None,
        start_of_sentence=None, end_of_sentence=None,
        pre_word_num=0, post_word_num=0,
        pre_surplus_idx=0, post_surplus_idx=1,
        use_short_concat=False,
        save_alignment=False,
        librosa_trim=False, attention_trim=False,
        time_str=None, isKorean=True):

    idx, (wav, alignment, path, text, sequence,mel) = args


    if use_short_concat:
        wav = short_concat(
                wav, alignment, text,
                start_of_sentence, end_of_sentence,
                pre_word_num, post_word_num,
                pre_surplus_idx, post_surplus_idx)

    if attention_trim and end_of_sentence:
        # attention이 text의 마지막까지 왔다면, 그 뒷부분은 버린다.
        end_idx_counter = 0
        attention_argmax = alignment.argmax(0)   # alignment: text length(encoder), target length(decoder)   ==> target length(decoder)
        end_idx = min(len(sequence) - 1, max(attention_argmax))
        max_counter = min((attention_argmax == end_idx).sum(), 5)

        for jdx, attend_idx in enumerate(attention_argmax):
            if len(attention_argmax) > jdx + 1:
                if attend_idx == end_idx:
                    end_idx_counter += 1

                if attend_idx == end_idx and attention_argmax[jdx + 1] > end_idx:
                    break

                if end_idx_counter >= max_counter:
                    break
            else:
                break

        spec_end_idx = hparams.reduction_factor * jdx + 3
        wav = wav[:spec_end_idx]
        mel = mel[:spec_end_idx]

    audio_out = inv_linear_spectrogram(wav.T,hparams)

    if librosa_trim and end_of_sentence:
        yt, index = librosa.effects.trim(audio_out, frame_length=5120, hop_length=256, top_db=50)
        audio_out = audio_out[:index[-1]]
        mel = mel[:index[-1]//hparams.hop_size]

    return audio_out

def get_most_recent_checkpoint(checkpoint_dir, checkpoint_step=None):
    if checkpoint_step is None:
        checkpoint_paths = [path for path in glob("{}/*.ckpt-*.data-*".format(checkpoint_dir))]
        idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]

        max_idx = max(idxes)
    else:
        max_idx = checkpoint_step
    lastest_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-{}".format(max_idx))
    print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    return lastest_checkpoint

def short_concat(
        wav, alignment, text,
        start_of_sentence, end_of_sentence,
        pre_word_num, post_word_num,
        pre_surplus_idx, post_surplus_idx):

    # np.array(list(decomposed_text))[attention_argmax]
    attention_argmax = alignment.argmax(0)

    if not start_of_sentence and pre_word_num > 0:
        surplus_decomposed_text = decompose_ko_text("".join(text.split()[0]))
        start_idx = len(surplus_decomposed_text) + 1

        for idx, attend_idx in enumerate(attention_argmax):
            if attend_idx == start_idx and attention_argmax[idx - 1] < start_idx:
                break

        wav_start_idx = hparams.reduction_factor * idx - 1 - pre_surplus_idx
    else:
        wav_start_idx = 0

    if not end_of_sentence and post_word_num > 0:
        surplus_decomposed_text = decompose_ko_text("".join(text.split()[-1]))
        end_idx = len(decomposed_text.replace(surplus_decomposed_text, '')) - 1

        for idx, attend_idx in enumerate(attention_argmax):
            if attend_idx == end_idx and attention_argmax[idx + 1] > end_idx:
                break

        wav_end_idx = hparams.reduction_factor * idx + 1 + post_surplus_idx
    else:
        if True: # attention based split
            if end_of_sentence:
                end_idx = min(len(decomposed_text) - 1, max(attention_argmax))
            else:
                surplus_decomposed_text = decompose_ko_text("".join(text.split()[-1]))
                end_idx = len(decomposed_text.replace(surplus_decomposed_text, '')) - 1

            while True:
                if end_idx in attention_argmax:
                    break
                end_idx -= 1

            end_idx_counter = 0
            for idx, attend_idx in enumerate(attention_argmax):
                if len(attention_argmax) > idx + 1:
                    if attend_idx == end_idx:
                        end_idx_counter += 1

                    if attend_idx == end_idx and attention_argmax[idx + 1] > end_idx:
                        break

                    if end_idx_counter > 5:
                        break
                else:
                    break

            wav_end_idx = hparams.reduction_factor * idx + 1 + post_surplus_idx
        else:
            wav_end_idx = None

    wav = wav[wav_start_idx:wav_end_idx]

    if end_of_sentence:
        wav = np.lib.pad(wav, ((0, 20), (0, 0)), 'constant', constant_values=0)
    else:
        wav = np.lib.pad(wav, ((0, 10), (0, 0)), 'constant', constant_values=0)

class Watcher:
    RASP_DIR = r'\\raspberrypi\pi\ai-makers-kit'
    WORK_DIR = r'C:\Users\jeyb1\workspace\Tacotron2-Wavenet-Korean-TTS'
    DIRECTORY_WATCH = r'\\raspberrypi\pi\ai-makers-kit\transformer_result' #폴더 경로 지정

    def __init__(self):
        self.observer = Observer()  #실행시 Observer객체 생성
  
    def run(self):
        os.chdir(self.RASP_DIR)
        event_handler = Handler() #이벤트 핸들러 객체 생성
        print("Hi handler!")
        self.observer.schedule(event_handler, self.DIRECTORY_WATCH, recursive=False)
        # self.observer.schedule(event_handler, r'.\text_test', recursive=True) #for local laptop 
        self.observer.start()
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
        rasp_dir_w = r'\\raspberrypi\pi\ai-makers-kit\transformer_result\answer.txt'
        work_dir = r'C:\Users\username\workspace\Tacotron2-Wavenet-Korean-TTS'
        print("Context changed!")
        
        time.sleep(1)
        f = open(rasp_dir_w, 'rt', encoding='euc-kr')
        # f = open(r'.\text_test\content.txt', 'r') # for local laptop
        text = f.read()
        f.close()

        splited = re.split(r'[.?!]', text)
        temp = []
        for i in range(len(text)):
            if text[i] == '.' or text[i] == '?' or text[i] == '!':
                temp.append(text[i])

        if len(splited) > 2 :
            text1 = splited[0]+ temp[0]
            text2 = splited[-2] + '.'

        text_list = [text1, text2]
        
        wav1 = synthesizer.synthesize(texts=text1, base_path="../generate",speaker_ids=[1],
                            attention_trim=True, base_alignment_path=None, isKorean=True)[0]
        wav2 = synthesizer.synthesize(texts=text2, base_path="../generate",speaker_ids=[1],
                            attention_trim=True, base_alignment_path=None, isKorean=True)[0]
        
        z_pad = np.zeros(10000)
        wav1 = np.append(wav1, z_pad)
        final_wav = np.append(wav1, wav2)
        
        rasp_dir = r'\\raspberrypi\pi\ai-makers-kit\tts_result'
        current_path = "answer.wav"
        
        os.chdir(rasp_dir)
        save_wav(final_wav, current_path, hparams.sample_rate)
        os.chdir(work_dir)

        print("synthesize finished!")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', required=True)
    parser.add_argument('--sample_path', default="../generate")
    # parser.add_argument('--text', required=True)
    parser.add_argument('--num_speakers', default=2, type=int)
    parser.add_argument('--speaker_id', default=1, type=int)
    parser.add_argument('--checkpoint_step', default=None, type=int)
    parser.add_argument('--is_korean', default=True, type=str2bool)
    parser.add_argument('--base_alignment_path', default=None)
    config = parser.parse_args()

    synthesizer = Synthesizer()
    synthesizer.load(config.load_path, config.num_speakers, config.checkpoint_step,inference_prenet_dropout=False)

    w = Watcher()
    w.run()
















