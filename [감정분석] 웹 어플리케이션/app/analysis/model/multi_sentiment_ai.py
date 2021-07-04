import argparse
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from app.analysis.model.dataset import LABEL_DICT, get_data_loader
from app.analysis.model.model import MultimodalTransformer
from flask.globals import g


def test(model,
             data_loader,
             device):
    
    y_pred = []

    model.eval()
    model.zero_grad()
    iterator = tqdm(enumerate(data_loader), desc='test_steps', total=len(data_loader))
    for step, batch in iterator:
        with torch.no_grad():
            
            # unpack and set inputs
            batch = map(lambda x: x.to(device) if x is not None else x, batch)
            audios, a_mask, texts, t_mask = batch
            
            # labels = labels.squeeze(-1).long()
            # y_true += labels.tolist()

            # feed to model
            logit, hidden = model(audios, texts, a_mask, t_mask)
            y_pred += logit.max(dim=1)[1].tolist()

    return logit, y_pred


def main(args):
    data_loader = get_data_loader(
        args=args,
        data_path=args.data_path,
        bert_path=args.bert_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        split=args.split
    )

    model = MultimodalTransformer(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_classes=args.n_classes,
        only_audio=args.only_audio,
        only_text=args.only_text,
        d_audio_orig=args.n_mfcc,
        d_text_orig=768,  # BERT hidden size
        d_model=args.d_model,
        attn_mask=args.attn_mask
    ).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    # Test
    model.zero_grad()
    logit, emotion_pred = test(model, data_loader, args.device)
    
    #print("model ouput(logit): \n", logit)
    #print("logit(tensor) size: ", logit.size())

    nplogit = logit.cpu().numpy()
    sample_num = len(nplogit) # data num
    total = np.zeros(7) # emotion num
    for sample in range(sample_num):
        #logit_1d = logit[sample, :] #2d to 1d
        nplogit_1d = nplogit[sample, :]
        #print("nplogit_1d shape: ", nplogit_1d.shape)
    
        # min = torch.min(logit_1d)
        # max = torch.max(logit_1d)
        min = np.min(nplogit_1d)
        max = np.max(nplogit_1d)

        nplogit_1d = (nplogit_1d-min)/(max-min)
        for element in range(len(nplogit_1d)):
            total[element] += nplogit_1d[element]
            
        # for element in range(len(nplogit_1d)):
        #     nplogit[element] = (nplogit_1d-min)/(max-min)
        #     total[element] += nplogit[element]
        #print("({}/{}) normalized logit: \n".format(sample+1, sample_num), nplogit_1d)

    total = total / sample_num

    # g.emotionAvg = total
    return total
    #print(total)
    #print("emotion_pred: ", emotion_pred)

def argu():
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--split', type=str, default='predict')
    parser.add_argument('--only_audio', action='store_true')
    parser.add_argument('--only_text', action='store_true')
    parser.add_argument('--data_path', type=str, default='./app/analysis')
    parser.add_argument('--bert_path', type=str, default='./KoBERT')
    parser.add_argument('--model_path', type=str, default='./app/analysis/model/epoch10-loss0.0292-f11.0000.pth')
    parser.add_argument('--n_classes', type=int, default=7)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=10)

    # architecture
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=40)
    parser.add_argument('--n_heads', type=int, default=10)
    parser.add_argument('--attn_mask', action='store_false')

    # data processing
    parser.add_argument('--max_len_audio', type=int, default=400)
    parser.add_argument('--sample_rate', type=int, default=48000)
    parser.add_argument('--resample_rate', type=int, default=16000)
    parser.add_argument('--n_fft_size', type=int, default=400)
    parser.add_argument('--n_mfcc', type=int, default=40)

    # args_ = parser.parse_args()
    args_, unknown = parser.parse_known_args()

    # -------------------------------------------------------------- #
    
    # check usage of modality
    if args_.only_audio and args_.only_text:
        raise ValueError("Please check your usage of modalities.")

    # seed and device setting
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args_.device = device_

    return main(args_)