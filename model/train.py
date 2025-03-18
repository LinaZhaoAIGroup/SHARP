import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm  import tqdm, trange

from glob import glob
from skimage import io
from addict import Dict
from IPython import display
import matplotlib.pyplot as plt

def train(train_dataloader, valid_dataloader, model, loss_fun, optimizer, schedule, epochs, device,early_stopping,patch_size):
    res = Dict(best_model=None, best_loss=None, train_losslist=[],val_losslist=[])
    model = model.to(device)
    for i, epoch in enumerate(trange(epochs)):
        try:
            model.train()
            epoch_loss = []
            iters=0
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            for img, tar in train_dataloader:
                if patch_size:
                    img = img.view(-1, 1, img.size()[-1], img.size()[-1])
                    tar = tar.view(-1, 1, tar.size()[-1], tar.size()[-1])
                #print(img.size())
                pre1= model(img)
            
                loss = loss_fun(pre1, tar)#+loss_fun(out,denosingtar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                #print('epoch {}/{} iters {} - 训练集loss: {}'.format(epoch, epochs,iters, loss))
                iters+=1
            epoch_loss = np.mean(epoch_loss)
            res.train_losslist.append(epoch_loss)
            with torch.no_grad():
                model.eval()
                val_loss=[]
                for img, tar, _ in valid_dataloader:
                    pre = model(img)
                    loss = loss_fun(pre, tar)
                    val_loss.append(loss.item())
                val_loss = np.mean(val_loss)
                res.val_losslist.append(val_loss)
            schedule.step(val_loss)
            
            if res.best_model == None or val_loss<res.best_loss:
                res.best_model, res.best_loss = model, val_loss
            else:pass

            print('epoch {}/{} - 训练集loss: {:.6f} - 验证集loss: {:.6f}'.format(epoch+1, epochs, epoch_loss,val_loss))
            early_stopping(val_loss, model)
         
            if early_stopping.early_stop:
                print("此时早停！")
                break
            if (i+1) % 10 ==0:
                plt.cla()
                plt.plot(res.train_losslist, color='black',label='train_loss')
                plt.plot(res.val_losslist, color='red',label='val_loss')
                plt.legend() 
                plt.xlim(-2, epoch+1)  
                display.clear_output(wait=True)
                plt.pause(0.00000001) 
                plt.show()
        except KeyboardInterrupt: 
            break
    plt.close()
    return res