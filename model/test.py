import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from skimage import io
from .function import rmse
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

def test(args,valid_dataloader,model1,num_projections):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1).to(args.device)
    loss_fun=torch.nn.MSELoss()

    MSE1=[]
    PSNR1=[]
    SSIM1=[]
    RMSE1=[]
    times=[]
    ms_ssim1=[]

    with torch.no_grad():
        for img, tar, file in valid_dataloader:
            try:
                import time
                s=time.time()
                pre = model1(img)
                t=time.time()-s
                print(t)
                ms_ssim0 = ms_ssim(pre,tar)
                MSE0= loss_fun(pre,tar)
                img=img.cpu().detach().numpy().squeeze().squeeze()
                pre=pre.cpu().detach().numpy().squeeze().squeeze()
                pre[pre>1]=1
                tar=tar.cpu().detach().numpy().squeeze().squeeze()

                PSNR0 =PSNR(tar,pre)
                SSIM0 = SSIM(tar,pre,data_range=2)
                RMSE0= rmse(tar,pre)
                
                times.append(t)
                MSE1.append(MSE0)
                PSNR1.append(PSNR0)
                SSIM1.append(SSIM0)
                RMSE1.append(RMSE0)
                ms_ssim1.append(ms_ssim0)

                
                plt.figure(figsize=(15,10))
                plt.subplot(2, 3, 1)
                plt.imshow(tar,cmap='gray')
                plt.axis('off')
                plt.title('tar')
                #plt.colorbar(shrink=0.5)
                

                plt.subplot(2, 3, 2)
                plt.imshow(pre,cmap='gray')
                plt.axis('off')
                plt.title('Pre')
                plt.xlabel("PSNR: {:.4f}  RMSE: {:.4f}\nssim: {:.4f}  ms_ssim: {:.4f}  MSE: {:.4f}".format(PSNR0,RMSE0,SSIM0,ms_ssim0,MSE0,fontsize=20))
                #plt.colorbar(shrink=0.5)
                
                
                plt.subplot(2, 3,3)
                plt.imshow(tar-pre,cmap="gray")
                plt.axis('off')
                plt.title('Pre-Tar')
                #plt.colorbar(shrink=0.5)

                plt.subplot(2, 3, 4)
                plt.imshow(tar[450:600,450:600],cmap='gray',vmax=1,vmin=0)
                plt.axis('off')

                plt.subplot(2, 3, 5)
                plt.imshow(pre[450:600,450:600],cmap='gray',vmax=1,vmin=0)
                plt.axis('off')

                plt.subplot(2, 3, 6)
                plt.plot(tar[512],color="r",label="Tar")
                plt.plot(pre[512],color="g",label="Pre")
                plt.legend(ncol=2,loc=1)

                #io.imsave("/home/zhengmao/hbtask/SHARP/data/bamboo_data/result/recon_"+os.path.basename(file[0]).split("_")[-1],pre)
                display.clear_output(wait=True)
                plt.pause(0.1)

            except KeyboardInterrupt:
                break
        PSNR1 = np.mean(PSNR1)
        SSIM1 = np.mean(SSIM1)
        RMSE1 = np.mean(RMSE1)
        time1 = np.mean(times)
    list0=[]
    for i in MSE1:
        i = i.to("cpu")
        list0.append(i)
    list1=[]
    for i in ms_ssim1:
        i = i.to("cpu")
        list1.append(i)
    print('Rmse:',np.mean(RMSE1),'PSNR:',np.mean(PSNR1),'SSIM:',np.mean(SSIM1),'MSE:',np.mean(list0),'MS_SSIM:',np.mean(list1),"m-time",time1)