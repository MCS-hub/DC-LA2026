import os
import torch
from torchvision import utils
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
import config


###
###Base class for neural networks for training/comparing
###

class baseNet():
    def __init__(self,args,path,net,data_loaders):
        self.args=args
        self.alg=args.alg
        self.net=net
        self.load_net(path)
        self.data_valid_loader, self.data_test_loader = data_loaders
        self.nograd=True
        print('Noise std:{}'.format(args.noise))
        print(f'No of Parameters: {sum(p.numel() for p in self.net.parameters())}')
        if(self.args.cuda):net.cuda()

    def load_net(self,name):
        print('loading net ...')
        print('os.path.isfile(name)', name,  os.path.isfile(name))
        if (os.path.isfile(name)):
            self.net.load_state_dict(torch.load(name))
            print('Loaded {} from checkpoint'.format(name))
        else:
            pass

    def save_img(self,name,img,save_pt=False):
        with torch.no_grad():
            utils.save_image(
                img.data,
                config.data_path+'figs/'+self.args.alg+'/'+str(self.args.expir)+'/'+name+'.png',
                normalize=True,
                nrow=10,
                value_range=(0, 1),
            )
            if save_pt:
                img = img.clone().detach().cpu()
                torch.save(img,config.data_path+'figs/'+self.args.alg+'/'+str(self.args.expir)+'/'+name+'.pt')

    def validate(self):
        if hasattr(self,'net'): self.net.eval()
        if(self.nograd):
            with torch.no_grad(): self.validate_cycle()
        else: self.validate_cycle()

    def ssim(self,output,truth):
        avssim = 0
        for i in range(truth.shape[0]):
            for j in range(truth.shape[1]):
                avssim += ssim_sk(truth[i,j].cpu().detach().numpy(),output[i,j].cpu().detach().numpy(),data_range=truth[i,j].max().detach().item()-truth[i,j].min().detach().item())
        return avssim/(truth.shape[0]*truth.shape[1])

    def psnr(self,output,truth):
        avpsnr = 0
        for i in range(truth.shape[0]):
            for j in range(truth.shape[1]):
                avpsnr += psnr_sk(truth[i,j].cpu().detach().numpy(),output[i,j].cpu().detach().numpy(),data_range=truth[i,j].max().detach().item()-truth[i,j].min().detach().item())
        return avpsnr/(truth.shape[0]*truth.shape[1])





