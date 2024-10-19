import re
import os
import argparse
from string import punctuation

import torch
import yaml
import librosa
import datetime
import time

import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
import python_speech_features as psf


from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

from pymcd.mcd import Calculate_MCD
from pesq import pesq
from scipy.io import wavfile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##################### Evaluation Metric #####################
#mcd, pesq, cer

def calculate_mcd_value(ref_wav_path, synth_wav_path):
    mcd_calculator = Calculate_MCD(MCD_mode="plain")
    # Calculate MCD
    try:
        mcd_value = mcd_calculator.calculate_mcd(ref_wav_path, synth_wav_path)
    except Exception as e:
        print(f"Error computing MCD: {e}")
        mcd_value = None
    return mcd_value

def calculate_pesq(ref_wav_path, synth_wav_path):
    # Load reference and synthesized signals
    ref_rate, ref_signal = wavfile.read(ref_wav_path)
    synth_rate, synth_signal = wavfile.read(synth_wav_path)
    
    # Convert to float32
    ref_signal = ref_signal.astype(np.float32)
    synth_signal = synth_signal.astype(np.float32)
    
    # Resample to 16kHz or 8kHz based on sample rate
    if ref_rate not in [8000, 16000]:
        #print(f"Unsupported reference sample rate: {ref_rate}. Resampling to 16kHz.")
        ref_signal = librosa.resample(ref_signal, orig_sr=ref_rate, target_sr=16000)
        ref_rate = 16000
    if synth_rate != ref_rate:
        #print(f"Sample rates do not match. Resampling synthesized signal from {synth_rate}Hz to {ref_rate}Hz.")
        synth_signal = librosa.resample(synth_signal, orig_sr=synth_rate, target_sr=ref_rate)
        synth_rate = ref_rate
        
    # Ensure same length
    min_len = min(len(ref_signal), len(synth_signal))
    ref_signal = ref_signal[:min_len]
    synth_signal = synth_signal[:min_len]
    
    # Determine mode based on sample rate
    if ref_rate == 8000:
        mode = 'nb'  # Narrowband
    elif ref_rate == 16000:
        mode = 'wb'  # Wideband
    else:
        print(f"Unsupported sample rate: {ref_rate}. PESQ supports 8kHz or 16kHz only.")
        return None

    try:
        pesq_score = pesq(ref_rate, ref_signal, synth_signal, mode)
    except Exception as e:
        print(f"Error computing PESQ: {e}")
        pesq_score = None
    return pesq_score
    

#log
def create_log_file():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"evaluation_log_{timestamp}.txt"
    return log_filename

def log_data(log_filename, data):
    with open(log_filename, "a") as file:
        file.write(data + "\n")


###############################################################
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




def synthesize(model, step, configs, vocoder, batchs, control_values, log_filename):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    total_pesq = 0
    total_mcd = 0
    num_samples=0
    log_filename = create_log_file()
    
    result_path = os.path.join(train_config["path"]["result_path"], str(step))
    os.makedirs(result_path, exist_ok=True)
    
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            
            #start = time.time()
            
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            
            # end = time.time()
            # inf_time = end-start
            
            
            synth_paths = synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                result_path,
            )
            
            # After synthesizing, compute PESQ and MCD
            ids = batch[0]
            for idx, id_ in enumerate(ids):
                # Paths to synthesized and reference wav files
                synth_wav_path = os.path.join(result_path, f"{id_}.wav")
                ref_wav_path = os.path.join(
                    preprocess_config["path"]["raw_path"], f"{id_}.wav"
                )
                
                if os.path.exists(synth_wav_path) and os.path.exists(ref_wav_path):
                    mcd_value = calculate_mcd_value(ref_wav_path, synth_wav_path)
                    pesq_score = calculate_pesq(ref_wav_path, synth_wav_path)
                    
                    
                    if pesq_score is not None and mcd_value is not None:
                        total_pesq += pesq_score
                        total_mcd += mcd_value
                    
                        num_samples += 1
                        log_data(log_filename, f"ID: {id_}, MCD: {mcd_value}, PESQ: {pesq_score}")
                    else:
                        log_data(log_filename, f"ID: {id_}, Fail to calculate evaluation metric")
                else:
                    print(f"No ID {id_}.")
                    log_data(log_filename, f"ID: {id_}, No file")
                    
    if num_samples > 0:
        average_pesq = total_pesq / num_samples
        average_mcd = total_mcd / num_samples
        
        log_data(log_filename, f"Average MCD: {average_mcd}, Average PESQ: {average_pesq}")
        print(f"Average MCD: {average_mcd}, Average PESQ: {average_pesq}")
    else:
        print("No File to calculate evaluation metric.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
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
        batchs = DataLoader(
            dataset,
            batch_size=8,
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
    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values, log_filename)