import argparse
import os
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import random
import numpy as np
import time

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def main(args, configs, seed):
    print(f"Prepare training for Seed {seed} ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    data_dir = '/userHome/userhome2/dahyun/FastSpeech2_SingleTTS/preprocessed_data/LJSpeech_paper/seed_all'
    train_data_file = os.path.join(data_dir, f'train_seed_{seed}.txt')
    val_data_file = os.path.join(data_dir, f'val_seed_{seed}.txt')
 
    # Train and Validation Datasets
    train_dataset = Dataset(train_data_file, preprocess_config, train_config, sort=True, drop_last=True)
    val_dataset = Dataset(val_data_file, preprocess_config, train_config, sort=False, drop_last=False)
        
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    
    # DataLoader for train and validation sets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
        
    train_log_path = os.path.join(train_config["path"]["log_path"], f"train_seed_{seed}")
    val_log_path = os.path.join(train_config["path"]["log_path"], f"val_seed_{seed}")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    # save_step = train_config["step"]["save_step"]
    # synth_step = train_config["step"]["synth_step"]
    # val_step = train_config["step"]["val_step"]

    # Config에서 성능 확인할 스텝 목록 가져오기
    specific_steps = train_config["step"]["specific_steps"]
    
    outer_bar = tqdm(total=total_step, desc=f"Training for Seed {seed}", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch} for Seed {seed}", position=1)
        for batchs in train_loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                # 특정 스텝에서만 성능 확인 및 모델 저장
                if step in specific_steps:
                    model.eval()
                    val_file = os.path.join(data_dir, f'val_seed_{seed}.txt')
                    message = evaluate(model, step, configs, val_file, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)
                        
               
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )


                
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            f"seed_{seed}_{step}.pth.tar",
                        ),
                    )
                    
                    model.train()

                if step == total_step:
                    return
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
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
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
            
    for seed in range(0,10):
        set_random_seed(seed)    
        
        start_time = time.time()
        log_file_path = os.path.join(train_config["path"]["log_path"], f"train_seed_{seed}", "train_log.txt")
        
        main(args, configs, seed)
        end_time = time.time()
        
        # 학습에 걸린 시간 계산
        elapsed_time = end_time - start_time
        
        # 시간을 시간, 분, 초로 변환
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_time_message = f"Seed {seed} Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"

        
        # 학습 시간이 로그 파일에 기록되도록 설정
        with open(log_file_path, "a") as log_file:
            log_file.write(elapsed_time_message)
        
        print(elapsed_time_message)
        
