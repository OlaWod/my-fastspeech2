import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from dataset import MyDataset
from evaluate import evaluate
from model import MyLoss, get_model
from utils import get_param_num, to_device


os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    preprocess_cfg, model_cfg, train_cfg = configs
    
    # dataset
    print("Loading dataset...")
    dataset = MyDataset(
        "train.txt", preprocess_cfg, train_cfg, sort=True, drop_last=True
    )

    batch_size = train_cfg["batch_size"]
    group_size = train_cfg["group_size"]
    assert batch_size * group_size < len(dataset)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    # model, optimizer
    print("Loading model...")
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    print("Number of model parameters:", get_param_num(model))

    # loss
    Loss = MyLoss().to(device)

    # output
    ckpt_dir = train_cfg["path"]["ckpt_dir"]
    log_dir = train_cfg["path"]["log_dir"]
    log_path = os.path.join(log_dir, "log.txt")
    val_path = os.path.join(log_dir, "log_val.txt")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # training
    step = train_cfg["optimizer"]["restore_step"] + 1
    epoch = 1

    grad_clip_thresh = train_cfg["optimizer"]["grad_clip_thresh"]

    total_step = train_cfg["step"]["total_step"]
    val_step = train_cfg["step"]["val_step"]
    log_step = train_cfg["step"]["log_step"]
    save_step = train_cfg["step"]["save_step"]

    print("Training...")
    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = train_cfg["optimizer"]["restore_step"]
    outer_bar.update()
    
    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for group in loader:
            for batch in group:
                batch = to_device(batch, device)

                # Forward
                output = model(*batch)

                # Cal loss
                loss = Loss(output, batch)
                total_loss = loss[0]

                # Backward
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                optimizer.update_lr_and_step()
                optimizer.zero_grad()

                # Log
                if step % log_step == 0:
                    loss = [l.item() for l in loss]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *loss
                    )
                    with open(log_path, "a") as f:
                        f.write(message1 + message2 + "\n")
                    outer_bar.write(message1 + message2)
                    
                # Eval
                if step % val_step == 0:
                    model.eval()
                    
                    message = evaluate(model, step, configs)
                    with open(val_path, "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                # Save
                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.optimizer.state_dict(),
                        },
                        os.path.join(
                            train_cfg["path"]["ckpt_dir"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                # Quit
                if step == total_step:
                    quit()

                step += 1
                outer_bar.update(1)
                
            inner_bar.update(1)
        epoch += 1
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p", "--preprocess_config", type=str, required=True, help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Load Config
    preprocess_cfg = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_cfg = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_cfg = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_cfg, model_cfg, train_cfg)

    main(args, configs) 
