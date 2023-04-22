import os
from typing import Dict
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms 
from torch.utils.data import DataLoader
from Diffusion.Model import UNet
import torch.optim as optim
from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Scheduler import GradualWarmupScheduler
from tqdm import tqdm
from torchvision.utils import save_image

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    #dataset
    dataset = CIFAR10(
        root = './CIFAR10', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),           
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    #model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mul=modelConfig["channel_mult"], attn=modelConfig["attn"], 
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
      
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, mulptiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    
    #start training
    for epoch in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataloader:
            for images, labels in tqdmDataloader:
                #train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataloader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "loss": loss.item(),
                    "img shape": x_0.shape,
                    "LR": optimizer.state_dict()["param_groups"][0]["lr"]
                })
        warmUpScheduler.step()
        if epoch % 10 == 0:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], f"ckpt_{epoch}_.pt"))    


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mul=modelConfig["channel_mult"], attn=modelConfig["attn"],
                         num_res_blocks=modelConfig["num_res_blocks"], dropout=0).to(device)
        ckpt = torch.load(os.path.join(modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print(f"model, load {modelConfig['test_load_weight']} done")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    # sampled from standard normal distribution
    noisyImage = torch.randn(
        size = [modelConfig["batch_size"], 3, 32, 32], device = device)
    saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
    save_image(saveNoisy, os.path.join(
        modelConfig["sample_dir"], modelConfig["sampledNoisyImgName"], nrow=modelConfig["nrow"]))
    sampledImgs = sampler(noisyImage)
    sampledImgs = sampledImgs * 0.5 + 0.5
    save_image(sampledImgs, os.path.join(
        modelConfig["sample_dir"], modelConfig["sampledImgName"], nrow=modelConfig["nrow"]))
