import torch
import config
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from data_load import create
import BaseAlg
from matplotlib import pyplot as plt

#ICNN
class convexnet(nn.Module):  # constant width
    def __init__(self, args, n_channels=16, kernel_size=5, n_layers=5, n_chan=1):
        super().__init__()
        self.args=args
        self.convex = True
        self.n_layers = n_layers
        self.leaky_relu = nn.LeakyReLU(negative_slope=.2)
        self.smooth_length=0
        # these layers can have arbitrary weights
        self.wxs = nn.ModuleList([nn.Conv2d(n_chan, n_channels, kernel_size=kernel_size, stride=1, padding=2, bias=True) for _ in range(self.n_layers+1)])
        # these layers should have non-negative weights
        self.wzs = nn.ModuleList([nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=2, bias=False) for _ in range(self.n_layers)])
        self.final_conv2d = nn.Conv2d(n_channels, 1, kernel_size=kernel_size, stride=1, padding=2, bias=False)

        self.initialize_weights()

    def initialize_weights(self, min_val=0, max_val=1e-3):
        for layer in range(self.n_layers):
            self.wzs[layer].weight.data = min_val + (max_val - min_val) * torch.rand_like(self.wzs[layer].weight.data)
        self.final_conv2d.weight.data = min_val + (max_val - min_val) * torch.rand_like(self.final_conv2d.weight.data)

    def clamp_weights(self):
        for i in range(self.smooth_length,self.n_layers):
            self.wzs[i].weight.data.clamp_(0)
        self.final_conv2d.weight.data.clamp_(0)

    def forward(self, x):
  
        if self.convex:
            self.clamp_weights()
        z = self.leaky_relu(self.wxs[0](x))
      
        for layer_idx in range(self.n_layers):
            z = self.leaky_relu(self.wzs[layer_idx](z) + self.wxs[layer_idx+1](x))
        z = self.final_conv2d(z)
        net_output = z.view(z.shape[0], -1).mean(dim=1,keepdim=True)
        assert net_output.shape[0] == x.shape[0], f"{net_output.shape}, {x.shape[0]}"
        return net_output

      
class MyNet(nn.Module):
    def __init__(self,args):
        super(MyNet, self).__init__()
        self.convnet2 = convexnet(args,n_channels=args.n_channels,n_chan=1,n_layers=args.n_layers)
        self.convnet=convexnet(args,n_channels=args.n_channels ,n_chan=1,n_layers=args.n_layers)

    def forward(self, image):
        image = image.to(torch.float32)
         
        output = self.convnet(image) - self.convnet2(image)
        return output
    
    def calculate_net2_grad(self, x):
        with torch.enable_grad():
            x.requires_grad = True
            y = self.convnet2(x)
            grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0]
        x.requires_grad = False
        grad = grad.detach()
        return grad   
    
    def calculate_net1_grad(self, x):
        with torch.enable_grad():
            x.requires_grad = True
            y = self.convnet(x)
            grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0]
        x.requires_grad = False
        grad = grad.detach()
        return grad

### The network with name MyNet is used by default
class Algorithm(BaseAlg.baseNet):
    def __init__(self,args,data_loaders,path=config.data_path+'nets/'):

        super(Algorithm, self).__init__(args,path,MyNet(args),data_loaders)
        self.args=args
        if args.lamb == 0:
            self.lamb=self.lamb_approx()
        else:
            self.lamb=args.lamb
        self.nograd=False
        self.cntr=1


    def lamb_approx(self):
        if (len(self.data_valid_loader) != 0):
            for i, (scans, truth, _) in enumerate(self.data_valid_loader):
                if (scans.nelement() == 0):
                    scans = create(truth, seed=self.args.seed)
                if i ==1: break
        else:
            for i, (scans, truth, _) in enumerate(self.data_test_loader):
                if (scans.nelement() == 0):
                    scans = create(truth, seed=self.args.seed)
                if i ==1: break

        if(config.angles!=0):gradient_truth = config.fwd_op_adj_mod((config.fwd_op_mod(truth)-scans))
        else:gradient_truth=truth-scans
        tmp = np.sqrt(np.sum(np.square(gradient_truth.numpy()), axis=(1, 2, 3)))
        print('Gradient norm samples:', tmp)
        lambdy = np.mean(np.sqrt(np.sum(np.square(gradient_truth.numpy()), axis=(1, 2, 3))))
        return lambdy

    def output_PSM(self,scans,truth=None, name=''):
        if truth is not None:
            self.save_img(f'groundtruth_{name}',truth.detach())

        guess = config.fbp_op_mod(scans)
        guess = torch.zeros_like(guess)  # start from zero image
        lambdas=self.lamb

        grad = torch.zeros(guess.shape).type_as(guess)
        guess=torch.nn.Parameter(guess)
        optimizer = torch.optim.SGD([guess], lr=self.args.eps, momentum=0.5)

        self.cntr+=1

        for j in range(self.args.iterates):

            if(truth is not None):
                loss = nn.MSELoss()(guess.detach(),truth.detach().cuda())
                ssim = self.ssim(guess.detach(),truth.detach())
                psnr = self.psnr(guess.detach(),truth.detach())
                print(j)
                print('MSE Loss:', loss.item())
                print('SSIM:',ssim)
                print('PSNR:',psnr)
                if (j + 1) % 1000 == 0:
                    self.save_img(f'PSM_Iter_{str(j+1).zfill(6)}_ssim{ssim:.4f}_psnr{psnr:.4f}_{name}', guess.detach(), save_pt=True)
            
            xk = guess.clone().detach()  # here to fix xk -- an anchor point

            temp = self.net.calculate_net2_grad(xk)

            for _ in range(self.args.K):
                optimizer.zero_grad()

                data_misfit=config.fwd_op_mod(xk)-scans
                grad = config.fwd_op_adj_mod(data_misfit)

                gamma = self.args.gamma
                proxy = xk - gamma*(grad - lambdas * temp) 
                proxy = proxy.detach().requires_grad_(False)
                lossm=lambdas*(self.net.convnet(guess)).sum()
                b = proxy.shape[0]
                objective = lossm + (1/(2*self.args.gamma)*torch.norm((proxy-guess).reshape(b,-1), dim=1)**2).sum() 
                objective.backward()
                optimizer.step()  # update guess

        return guess

    def prox_net2(self, x, lam, gamma=None, n_iters=1):
        if gamma is None:
            gamma = lam
        u = x.clone()
        for _ in range(n_iters):
            grad_r2 = self.net.calculate_net2_grad(u)
            #u = u - gamma * (grad_r2 + (1/lam) * (u - x))
            u = x - gamma * grad_r2
        return u
    
    def prox_net1(self, x, lam, gamma=None, n_iters=1):
        if gamma is None:
            gamma = lam
        u = x.clone()
        for _ in range(n_iters):
            grad_r1 = self.net.calculate_net1_grad(u)
            #u = u - gamma * (grad_r1 + (1/lam) * (u - x))
            u = x - gamma * grad_r1
        return u


    def output_DCLA(self, scans, truth=None, name='', last_only=False, save=False, chain=0):
        x = config.fbp_op_mod(scans)   # X_0
        x = torch.zeros_like(x) # we want to explore more

        lambdas = self.lamb       
        tau = lambdas / self.args.noise**2

        def prox_r2(x, lam):
            return self.prox_net2(x, lam * tau)
        def prox_r1(v, lam):
            return self.prox_net1(v, lam * tau)

        gamma = self.args.gamma * self.args.noise**2 # rescale gamma 
        sm = self.args.sm                # smoothing parameter

        self.cntr += 1

        if not last_only:
            burn_in = self.args.burn_in
            assert burn_in < self.args.iterates, "Burn-in period must be less than total iterations."

        if truth is not None:
            self.save_img(f'GroundTruth_{name}', truth.detach())

        samples = []
        for j in range(self.args.iterates):

            # ---- data term gradient ∇f(X_k) ----

            data_misfit = config.fwd_op_mod(x) - scans
            grad_f = config.fwd_op_adj_mod(data_misfit)

            # scale
            grad_f = (1 / self.args.noise**2) * grad_f

            # ---- DC-LA update ----
            # Z_{k+1} ~ N(0, I)
            Z = torch.randn_like(x)

            # prox_{λ r2}(X_k)
            prox_r2_x = prox_r2(x, sm)

            # noise terms
            sqrt_2gamma = torch.sqrt(
                torch.tensor(2.0 * gamma, device=x.device, dtype=x.dtype)
            )

            # inner argument for prox_{(λ+γ) r1}
            inner = ((sm + gamma) / sm) * x \
                    - gamma * grad_f \
                    - (gamma / sm) * prox_r2_x \
                    + sqrt_2gamma * Z

            prox_r1_inner = prox_r1(inner, sm + gamma)

            # full DC-LA step
            x_next = x \
                - (gamma * sm) / (gamma + sm) * grad_f \
                - (gamma / (gamma + sm)) * prox_r2_x \
                + (sm * sqrt_2gamma) / (gamma + sm) * Z \
                + (gamma / (gamma + sm)) * prox_r1_inner

            x = x_next.detach()  # detach to avoid graph blow-up

            if not last_only and j >= burn_in:
                samples.append(x.cpu().numpy())
                if j % 20 == 0:
                    ssim = self.ssim(x.detach(), truth.detach())
                    psnr = self.psnr(x.detach(), truth.detach())
                    print(f'Iteration {j}: SSIM={ssim}, PSNR={psnr}')
                    self.save_img(f'Sample:{str(j).zfill(6)}_ssim{ssim:.4f}_psnr{psnr:.4f}_{name}',x)

            if save:
                if j % 20 ==0:
                    self.save_img(f'DCLA_Chain_{chain}_Iter_{str(j)}_{name}', x, save_pt=True)
        if not last_only:
            samples = np.array(samples)
            mean = np.mean(np.array(samples), axis=0)
            var = np.var(np.array(samples), axis=0)
            
            vmin = var.min()
            vmax = var.max()
            var_norm = (var - vmin) / (vmax - vmin + 1e-12)

            if truth is not None:
                diff_img = torch.abs(truth - torch.tensor(mean).type_as(x))
                dmin = diff_img.min()
                dmax = diff_img.max()
                diff_norm = (diff_img - dmin) / (dmax - dmin + 1e-12)
                self.save_img(f'DifferenceImage_{name}', diff_norm)

            self.save_img(f'VarianceImage_{name}', torch.tensor(var_norm).type_as(x))

            ssim = self.ssim(torch.tensor(mean).type_as(x), truth.detach())
            psnr = self.psnr(torch.tensor(mean).type_as(x), truth.detach())
            print(f'Mean Image SSIM: {ssim}, PSNR: {psnr}')
            self.save_img(f'MeanImage_ssim{ssim:.4f}_psnr{psnr:.4f}_{name}', torch.tensor(mean).type_as(x))
        

        return x

    def output_DCLA_multiple(self, scans, truth=None, chains=4, name=''):
        outputs = []
        for chain in range(chains):
            seed = 12345 + chain
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            print(f"Starting chain {chain+1}/{chains}")
            out = self.output_DCLA(scans, truth=truth, last_only=True, name=name, chain=chain)
            outputs.append(out.detach())
        
        outputs = torch.stack(outputs)  # shape: (chains, B, C, H, W)


        for j, img in enumerate(outputs):
            ssim = self.ssim(img, truth.detach())
            psnr = self.psnr(img, truth.detach())
            self.save_img(f'DCLA_Chain_Output_{j+1}_ssim{ssim:.4f}_psnr{psnr:.4f}_{name}', img, save_pt=True)
            print(f'Chain {j+1} Image SSIM: {ssim}, PSNR: {psnr}')
        final_output = torch.mean(outputs, dim=0)
        ssim = self.ssim(final_output, truth.detach())
        psnr = self.psnr(final_output, truth.detach())
        self.save_img(f'DCLA_MultipleChains_Mean_ssim{ssim:.4f}_psnr{psnr:.4f}_{name}', final_output)
        print(f'Multiple Chains Mean Image SSIM: {ssim}, PSNR: {psnr}')
        var_img = torch.var(outputs, dim=0)
        vmin = var_img.min()
        vmax = var_img.max()
        var_norm = (var_img - vmin) / (vmax - vmin + 1e-12)
        self.save_img(f'DCLA_MultipleChains_Variance_{name}', var_norm)

        abs_diff = torch.abs(truth - final_output)
        dmin = abs_diff.min()
        dmax = abs_diff.max()
        diff_norm = (abs_diff - dmin) / (dmax - dmin + 1e-12)
        self.save_img(f'DCLA_MultipleChains_error_{name}', diff_norm)

        return final_output

    def validate_cycle(self):
        all_ssim=0.0
        all_psnr=0.0
        avg_mse_loss=0.0
        print("Validate cycle...")
        data=self.data_test_loader

        for i, (scans, truth, name) in enumerate(data):
            name = name[0]  
            name = name[:-4]
            print(f'cycle: {i}/{len(data)}')
            if (scans.nelement() == 0):
                scans = create(truth, seed=self.args.seed)
            if self.args.cuda:
                scans,truth = scans.cuda(), truth.cuda()

            if self.args.test_mode == 'PSM':
                output = self.output_PSM(scans,truth, name=name)
            elif self.args.test_mode == 'DCLA':
                output = self.output_DCLA(scans,truth, name=name)
            elif self.args.test_mode == 'DCLA_mul':
                output = self.output_DCLA_multiple(scans,truth, chains=self.args.chains, name=name)
            else:
                raise ValueError('Invalid test mode')

            mse_loss = nn.MSELoss()(output,truth).detach().cpu().item()
            avg_ssim = self.ssim(output,truth)
            avg_psnr = self.psnr(output,truth)

            avg_mse_loss+=mse_loss
            all_ssim+=avg_ssim
            all_psnr+=avg_psnr

