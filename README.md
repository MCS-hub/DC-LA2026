# DC-LA: Difference-of-convex Langevin Algorithm


## $\ell_1 - \ell_2$ prior

To obtain histograms of the samples from the three samplers in the 2D experiment, run the following.
**Use multiple Markov chains and retain only the final samples from each chain.**
```bash
python l12_exp2.py
```
**Use one chain, discarding burn-in samples**
```bash
python l12_exp.py
```
**To produce binned KL plots for comparisons**
```bash
python l12_exp2_binKL.py
```
**To perform ablation experiments on (lambda,gamma)**
```bash
python ablation.py
```
## DCINNs prior

We leverage the ADCR repo: https://github.com/YasminZhang/ADCR and the pretrained DCINNs parameters (limited.pt) provided by Yasi Zhang.
The Mayo Grand Challenge data is from: https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h
(here we put some Mayo data inside the tomo\valid and tomo\test folders)


**To run DC-LA**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --alg=ADCR --iterates=2000 --noise=0.2 --seed=10 --load=./data/nets_new/ADCR/limited/limited.pt --test_mode=DCLA_mul --sm 1e-4 --chains=100 --ntest=1
```
Change `ntest` to access different CT scans.

**To run PSM for MAP estimation**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --eps=1e-6 --alg=ADCR --iterates=20000 --noise=0.2 --seed=10  --load=./data/nets_new/ADCR/limited/limited.pt --test_mode=PSM --ntest=1
```