import os
import torch
import numpy as np
from main_parser import parser
import config
torch.set_default_dtype(torch.float32)
from data_load import load_data
from tensorboardX import SummaryWriter


args = parser.parse_args()
config.init(args)
torch.manual_seed(args.seed)

def cntexpir():
    print(config.data_path+'logs/'+args.alg)
    if not os.path.exists(config.data_path+'logs'):
        os.makedirs(config.data_path+'logs')
    if (args.expir != -1):
        return args.expir
    exps = np.array([int(i[4:-4]) for i in (os.listdir(config.data_path+'logs/'))])
    expir=1
    if (len(exps) != 0):
        expir=exps.max()+1
    print('Experiment '+str(expir))
    args.expir=expir
    return expir


def main():
    data_loaders = load_data(args)
    print('Data loaded')
    cntexpir()
    if not os.path.exists(config.data_path+'figs/'+args.alg):
        os.makedirs(config.data_path+'figs/'+args.alg)
    if not os.path.exists(config.data_path+'figs/'+args.alg+'/'+str(args.expir)):
        os.makedirs(config.data_path+'figs/'+args.alg+'/'+str(args.expir))


    alglib = __import__(args.alg)
    Algorithm=alglib.Algorithm(args,data_loaders,path=args.load)

    print('mode:' + args.test_mode)

    # write to a txt file
    with open(config.data_path+'logs/'+args.alg+str(args.expir)+'.txt',"a") as f:
        f.write('mode:' + args.test_mode + '\n')
        f.write('noise: ' + str(args.noise) + '\n')
        f.write('Full parameters: '+str(args)+'\n')

    Algorithm.validate()


if __name__ == '__main__': main()
