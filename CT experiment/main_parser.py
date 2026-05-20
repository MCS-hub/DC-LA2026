import argparse




def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Training settings
parser = argparse.ArgumentParser(description='CT comparison framework')
# parser.add_argument('--batch-size', type=int, default=20, metavar='N',
#                     help='input batch size (default: 20)')
parser.add_argument('--noise', type=float, default=0.01, metavar='N',
                    help='noise level of the mean for the images (default: 0.01), i.e. 1 percent')
parser.add_argument('--seed', type=int, default=10, metavar='SD',
                    help='random seed (default: 10)')
parser.add_argument('--log_interval', type=int, default=2, metavar='L',
                    help='how many batches to wait before logging training status')
parser.add_argument('--iterates', type=int, default=10,
                    help='how many iterates for algorithms like LG or LPD/ For iterative schemes it is the number of steps for gradient descent')
parser.add_argument('--cp_interval', type=int, default=5000, metavar='C',
                    help='how many batches to wait before saving a checkpoint')
parser.add_argument('--data-path', type=str, default='/local/scratch/public/zs334/compFrm/',#default='/home/zs334/rds/hpc-work/CompFrm/',
                    help='path to the data')
parser.add_argument('--alg', type=str, default='compare',
                    help='FBP,TV,FBP+U,FL,LG,LPD,TV,CLAR')
parser.add_argument('--cuda', type=str2bool, default=True,
                    help='use CUDA for training?')
parser.add_argument('--outp', type=str2bool, default=True,
                    help='output intermideate results?')
parser.add_argument('--init', type=str2bool, default=True,
                    help='Initialise weights differntly?')
parser.add_argument('--load', type=str, default='', metavar='LD',
                    help='Load path?')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--detectors', default=700, type=int, metavar='DET',
                    help='number of detectors for Ray Transform')
parser.add_argument('--clamping', type=float, default=0, metavar='CLAMP',
                    help='Clamping value')
parser.add_argument('--eps', type=float, default=1e-5, metavar='EPS',
                    help='Value for constant eps for PSM')
parser.add_argument('--expir', type=int, default=-1, metavar='EXP',
                    help='Experiment number to run with')
parser.add_argument('--clip', type=float, default=1, metavar='CLP',
                    help='Clip value for the gradient')
parser.add_argument('--size', type=int, default=16, metavar='EXP',
                    help='Size as a parameter')
parser.add_argument('--mult', type=str2bool, default=False, metavar='MLT',
                    help='Multiple instances running? So different checkpoint loaders')
parser.add_argument('--wclip', type=str2bool, default=True, metavar='WCL',
                    help='Clip weights to 0?')
# add an argument test_only
# parser.add_argument('--test', type=str2bool, default=False, metavar='TST',
#                     help='Test only')
# add an argument setting=sparse or limited
# parser.add_argument('--setting', type=str, default='limited', metavar='SET',
#                     help='sparse or limited')
# add an argument alpha
parser.add_argument('--alpha', type=float, default=0.0, metavar='ALP',
                    help='alpha value for TV regularization')
# add an argument test_mode
parser.add_argument('--test_mode', type=str, default='PSM', metavar='TM',
                    help='test mode')
# add an argument K
parser.add_argument('--K', type=int, default=1, metavar='K',
                    help='K value for inner loop')  

# add an argument n_channels
parser.add_argument('--n_channels', type=int, default=32, metavar='NC',
                    help='number of channels')

# add an argument n_layers
parser.add_argument('--n_layers', type=int, default=10, metavar='NL',
                    help='number of layers')


# add a comment
parser.add_argument('--comment', type=str, default='', metavar='CMT', help='comment')


# add an argument momentum
parser.add_argument('--momentum', type=float, default=0.5, metavar='MOM', help='momentum')

# add an argument batch_size_valid
parser.add_argument('--batch_size_valid', type=int, default=1, metavar='BSV', help='batch size for validation')

# add an argument smoothing
parser.add_argument('--sm', type=float, default=1e-4, metavar='SM', help='smoothing parameter for DCLA')

# add an argument burn_in
parser.add_argument('--burn_in', type=int, default=0, metavar='BI', help='burn-in period for DCLA')

# add an argument chains
parser.add_argument('--chains', type=int, default=4, metavar='CHN', help='number of chains for DCLA multiple chains')

# add an argument init_noise
parser.add_argument('--init_noise', type=float, default=0.0, metavar='INITN', help='initialization noise level')

# add an argument gamma
parser.add_argument('--gamma', type=float, default=1e-5, metavar='GAM', help='step size parameter for ADCR')

# add an argument ntest
parser.add_argument('--ntest', type=int, default=2, metavar='NTEST', help='number of test samples to use')

# add an argument init_std
parser.add_argument('--init_std', type=float, default=0.0, metavar='INITS', help='standard deviation for init state')

# add an argument lamb
parser.add_argument('--lamb', type=float, default=0.0, metavar='LAMB', help='regularization parameter lambda for ADCR')

# add an argument print_console
parser.add_argument('--pr_console', type=str2bool, default=True, metavar='PC', help='print to console during testing')