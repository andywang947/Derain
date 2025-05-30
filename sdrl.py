import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

from PIL import Image as Image
from torch.nn import MSELoss
from torch.optim import Adam
import torch.nn.functional as F

from tqdm import tqdm

from utils import Timer
from network import UNet, ResNet, DnCNN


# from restormer import Restormer
from data import SDR_dataloader, train_dataloader
import copy


torch.manual_seed(3)
parser = argparse.ArgumentParser()
dataset = 'Rain12'
parser.add_argument("--rainy_data_path", type=str, default="./dataset/"+dataset+"/", help='Path to rainy data')
parser.add_argument("--sdr_data_path", type=str, default="./dataset/"+dataset+"/sdr/", help='Path to sdr data')
parser.add_argument("--result_path", type=str, default="./dataset/"+dataset+"/result_origin_retrain/", help='Path to save result')
parser.add_argument("--backbone", type=str, default="Unet", help= "select backbone to be used in SDRL")
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--mt", action="store_true", help="Use mean teacher student model")
parser.add_argument("--ema_decay", type=float, default=0.99)

opt = parser.parse_args()

data_path = opt.rainy_data_path
save_path = opt.result_path
sdr_path = opt.sdr_data_path
epochs = opt.epoch
mt = opt.mt # Teacher student model
ema_decay = opt.ema_decay

if mt :
    print("Use Mean Teacher Student Model")
else :
    print("Doesn't use mean teacher student model.")

def update_ema(teacher_model, student_model, ema_decay):
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data = ema_decay * teacher_param.data + (1 - ema_decay) * student_param.data

loss_function = MSELoss()
data_loader = train_dataloader(data_path, batch_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    os.makedirs(save_path)
    if mt is True :
        os.makedirs(save_path + "teacher/")
except:
    pass

epoch_timer = Timer('s') 
total_time = 0

print("Start Training")

for batch in data_loader:
    try:
        # train 
        rainy_images, __ , ___, name = batch

        # print(rainy_images.shape)

        # To solve the UNet dimension problem.

        h,w = rainy_images.shape[2], rainy_images.shape[3]
        factor = 16
        # print(h,w)
        # if h%factor!=0 or w%factor!=0 :
        #     print("the size 有問題！") 
        # else:
        #     print("size 沒問題！")

        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        rainy_images = F.pad(rainy_images, (0,padw,0,padh), 'reflect')

        # img = np.clip(rainy_images[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        # plt.imsave(os.path.join(save_path,"teacher" ,name[0]), img)
        # exit()
        # new_h,new_w = rainy_images.shape[2], rainy_images.shape[3]
        # print(new_h,new_w)

        img_save_path = os.path.join(save_path,name[0])
        print(img_save_path)
        if os.path.exists(img_save_path) == True :
            print("the image exists!")
            continue
        else :
            print("The image now is :", name[0])

        epoch_timer.tic()    
        
        if opt.backbone == "Unet":
            model = UNet()
        elif opt.backbone == "ResNet":
            model = ResNet()
        elif opt.backbone == "DnCNN":
            model = DnCNN()
        elif opt.backbone == "Restormer":
            model = Restormer()
        
        if mt :
            teacher_model = copy.deepcopy(model)  # 或用 load_state_dict
            teacher_model.to(device)
            print("now is using mean teacher student model !")
        else :
            print("now is not using mean teacher student model !")
        
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=0.001)

        inner_batch_size = 1
        rainy_images = rainy_images.to(device)
        model.train()
        SDR_loader = SDR_dataloader(os.path.join(sdr_path, name[0][:-4]), batch_size=inner_batch_size)
        for j in tqdm(range(epochs)):
            for k, inner_batch in enumerate(SDR_loader):
                sdr_images = inner_batch

                sdr_images = F.pad(sdr_images, (0,padw,0,padh), 'reflect')
                # continue


                sdr_images = sdr_images.to(device)
                if mt :
                    images = torch.cat([rainy_images for _ in range(len(sdr_images))],0)
                    student_output = model(images)
                    teacher_output = teacher_model(images)
                    consistency_loss = loss_function(student_output, teacher_output)
                    loss = loss_function(student_output, sdr_images) + consistency_loss
                else :
                    images = torch.cat([rainy_images for _ in range(len(sdr_images))],0)
                    net_output = model(images)
                    loss = loss_function(net_output, sdr_images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if mt is True :  # update EMA !
                    update_ema(teacher_model, model, ema_decay)
        
        # inference
        if len(SDR_loader) == 0 :
            print("SDR image doesn't exist !")
            continue
        model.eval()
        net_output = model(rainy_images)

        # net_output = net_output[:,:,:h,:w]

        time = epoch_timer.toc()
        print("Time: ", time)
        total_time += time
        net_output = net_output[:,:,:h,:w]
        denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        plt.imsave(os.path.join(save_path,name[0]), denoised)

        if mt is True :
            teacher_model.eval()
            teacher_net_output = teacher_model(rainy_images)
            teacher_net_output = teacher_net_output[:,:,:h,:w]
            denoised_teacher = np.clip(teacher_net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
            plt.imsave(os.path.join(save_path,"teacher" ,name[0]), denoised_teacher)
    
    except Exception as e:
        print("Exception occur: ", e)
        pass
    
print("Finish! Average Time:", total_time/len(data_loader))

# python sdrl.py --rainy_data_path="./dataset/Rain100L/" --sdr_data_path="./dataset/Rain100L/sdr/" --result_path="./dataset/Rain100L/result_test/" --mt