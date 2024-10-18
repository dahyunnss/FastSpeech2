import re
import argparse
from string import punctuation
import os

import librosa
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

import datetime
import time
from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

from pymcd.mcd import Calculate_MCD
from pesq import pesq
from scipy.io import wavfile

mcd_toolbox = Calculate_MCD(MCD_mode="plain")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_rtf(audio_length, processing_time):
    return processing_time / audio_length

def calculate_pesq(ref_wav_path, synth_wav_path):
    ref, ref_rate = librosa.load(ref_wav_path, sr=None)
    synth, synth_rate = librosa.load(synth_wav_path, sr=None)
    
    # Ensure both signals have the same sampling rate
    if ref_rate != synth_rate:
        raise ValueError("Reference and synthesized files have different sampling rates")
    
    # Resample if needed
    if ref_rate not in [8000, 16000]:
        ref = librosa.resample(ref, orig_sr=ref_rate, target_sr=16000)
        synth = librosa.resample(synth, orig_sr=synth_rate, target_sr=16000)
        ref_rate = synth_rate = 16000
    
    ref = np.asarray(ref * 32768, dtype=np.int16)  # Convert to int16 format
    synth = np.asarray(synth * 32768, dtype=np.int16)  # Convert to int16 format
    
    return pesq(ref_rate, ref, synth, 'wb')

def validate_audio_length(audio_path, min_duration=0.25):
    audio, sr = librosa.load(audio_path, sr=None)
    duration = len(audio) / sr
    return duration >= min_duration

def create_log_file():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"mcd_log_{timestamp}.txt"
    return log_filename

def log_data(log_filename, data):
    with open(log_filename, "a") as file:
        file.write(data + "\n")
        
def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def get_all_wav_files(directory):
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
                
    return sorted(wav_files)


def synthesize(model, step, configs, vocoder, data_loader, control_values, log_filename, synthesized_wav_dir, reference_wav_dir):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    mcd_scores = []
    pesq_scores = []
    inference_times = []
    rtf_scores = []
    
    for idx, batch in enumerate(data_loader):
        batch = to_device(batch, device)
    
        with torch.no_grad():
            start = time.time()
            
            # Forward
            output = model(
                *(batch[2:]), 
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            
            end = time.time()
            
            inf_time = end - start #gpu
            inference_times.append(inf_time)
            log_data(log_filename, f"Inference Time: {inf_time:} seconds")
            
            # Ensuring each batch generates new files by creating a unique directory for each batch
            batch_synthesized_dir = os.path.join(synthesized_wav_dir, f"batch_{idx+1}")
            os.makedirs(batch_synthesized_dir, exist_ok=True)
            
            print(f"Batch {idx+1}: Synthesizing samples...")
            
            wav_start = time.time()
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                batch_synthesized_dir, #train_config["path"]["result_path"],
            )
            wav_end = time.time()
            wav_gen_time = wav_end - wav_start #gpu 
            
            synthesized_files = sorted([f for f in os.listdir(batch_synthesized_dir) if f.endswith('.wav')])
            reference_files = get_all_wav_files(reference_wav_dir)
            
            for synthesized_file, reference_file in zip(synthesized_files, reference_files):
                synthesized_wav_path = os.path.join(batch_synthesized_dir, synthesized_file)
                reference_wav_path = reference_file
            
                if not validate_audio_length(synthesized_wav_path) or not validate_audio_length(reference_wav_path):
                    print(f"Audio file too short: {synthesized_file} or its reference")
                    continue

                # Calculate audio length
                audio, _ = librosa.load(synthesized_wav_path, sr=None)
                audio_length = len(audio) / 22050  # Assuming 22050 Hz sample rate

                # Calculate RTF
                total_time = inf_time + wav_gen_time
                rtf = calculate_rtf(audio_length, total_time)
                rtf_scores.append(rtf)
                
                # Calculate MCD and PESQ
                mcd_value = mcd_toolbox.calculate_mcd(reference_wav_path, synthesized_wav_path)
                mcd_scores.append(mcd_value)
                
                pesq_value = calculate_pesq(reference_wav_path, synthesized_wav_path)
                if pesq_value is not None:
                    pesq_scores.append(pesq_value)
                    log_data(log_filename, f"{synthesized_file}: MCD Score = {mcd_value:}, PESQ Score = {pesq_value:}, RTF = {rtf:}")
                else:
                    log_data(log_filename, f"{synthesized_file}: MCD Score = {mcd_value:}, PESQ Score = N/A, RTF = {rtf:}")            
                #log_data(log_filename, f"{synthesized_file}: MCD Score = {mcd_value:}, PESQ Score = {pesq_value:}, RTF_GPU = {rtf:}")

    if mcd_scores or pesq_scores or rtf_scores:
        average_mcd = sum(mcd_scores) / len(mcd_scores)
        average_pesq = sum(pesq_scores) / len(pesq_scores)
        avg_inf_time = sum(inference_times) / len(inference_times)
        average_rtf = sum(rtf_scores) / len(rtf_scores)
        log_data(log_filename, f"Average MCD Score: {average_mcd:}, Average PESQ Score: {average_pesq:}, Average RTF_GPU: {average_rtf:}")
        log_data(log_filename, f"Average Inference Time:: {avg_inf_time:} seconds")
            
            
    # for batch in batchs:
    #     batch = to_device(batch, device)
    #     with torch.no_grad():
    #         # Forward
    #         output = model(
    #             *(batch[2:]),
    #             p_control=pitch_control,
    #             e_control=energy_control,
    #             d_control=duration_control
    #         )
    #         synth_samples(
    #             batch,
    #             output,
    #             vocoder,
    #             model_config,
    #             preprocess_config,
    #             train_config["path"]["result_path"],
    #         )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"], #batch
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--synthesized_wav_dir",
        type=str,
        required=True,
        help="directory for synthesized wav files",
    )
    parser.add_argument(
        "--reference_wav_dir",
        type=str,
        required=True,
        help="directory for reference wav files",
    )
    
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        data_loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control
    log_filename = create_log_file()
    synthesize(model, args.restore_step, configs, vocoder, data_loader, control_values, log_filename, args.synthesized_wav_dir, args.reference_wav_dir)
