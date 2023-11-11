import whisper
import os
import json
import torchaudio
import argparse
import torch
from text2phone import *
import traceback

lang2token = {
    'zh': "ZH|",
    'ja': "JP|",
    "en": "EN|",
}


def transcribe_one(audio_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)
    return lang, result.text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="CJ")
    parser.add_argument("--whisper_size", default="medium")
    parser.add_argument("--data_dir", default="",
                        help="目录下面保存子目录，每个子目录为一个speaker名字，speaker目录下面是改speaker的所有音频。")
    parser.add_argument("--config_file", default="./configs/config.json")
    parser.add_argument("--save_meta_file", default="")
    parser.add_argument("--save_vocab_file", default="")
    args = parser.parse_args()
    if args.languages == "CJE":
        lang2token = {
            'zh': "ZH|",
            'ja': "JP|",
            "en": "EN|",
        }
    elif args.languages == "CJ":
        lang2token = {
            'zh': "ZH|",
            'ja': "JP|",
        }
    elif args.languages == "C":
        lang2token = {
            'zh': "ZH|",
        }
    assert (torch.cuda.is_available()), "Please enable GPU in order to run Whisper!"
    model = whisper.load_model(args.whisper_size)
    parent_dir = args.data_dir
    speaker_names = list(os.walk(parent_dir))[0][1]
    total_files = sum([len(files) for r, d, files in os.walk(parent_dir)])
    # resample audios
    # 2023/4/21: Get the target sampling rate
    with open(args.config_file, 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']
    processed_files = 0

    phone_list = []
    processed_line_list = []

    with open(args.save_meta_file + "_temp", 'w', buffering=1) as fw:
        for speaker in speaker_names:
            for i, wavfile in enumerate(list(os.walk(parent_dir + speaker))[0][2]):
                try:
                    save_path = parent_dir + speaker + "/" + f"processed_{i}.wav"

                    if not os.path.exists(save_path):
                        wav, sr = torchaudio.load(parent_dir + speaker + "/" + wavfile, frame_offset=0, num_frames=-1,
                                                  normalize=True,
                                                  channels_first=True)
                        wav = wav.mean(dim=0).unsqueeze(0)
                        if sr != target_sr:
                            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
                        if wav.shape[1] / sr > 20:
                            print(f"{wavfile} too long, ignoring\n")
                        torchaudio.save(save_path, wav, target_sr, channels_first=True)

                    # ----------------------
                    # transcribe text
                    # ----------------------
                    lang, text = transcribe_one(save_path)
                    if lang not in list(lang2token.keys()):
                        print(f"{lang} not supported, ignoring\n")
                        continue

                    # ----------------------
                    # 转换为phone
                    # ----------------------
                    phone = text2phone(text, language=args.languages.lower())
                    phone_list.append(phone)
                    w_line = f"{save_path}|{speaker}|{lang2token[lang]}{text}|{phone}\n"
                    processed_line_list.append(w_line)

                    fw.write(w_line)

                    processed_files += 1
                    print(f"Processed: {processed_files}/{total_files}")
                except Exception as e:
                    print(f"--error:{e}")
                    continue

    # -----------------------
    # save vocab
    # -----------------------
    save_all_vocab(phone_list, args.save_vocab_file)

    # -----------------------
    # phone2id
    # -----------------------
    vocab = load_vocab(args.save_vocab_file)

    with open(args.save_meta_file, 'w', buffering=1) as fw:
        for line in processed_line_list:
            phone = line.split("|")[-1]
            indices = vocab.lookup_indices(phone.split())
            indices_str = ' '.join([str(i) for i in indices])

            w_line = f"{line}|{indices_str}\n"
            fw.write(w_line)
