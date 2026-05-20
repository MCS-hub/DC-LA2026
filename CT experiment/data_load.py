from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import config
from torch.autograd import Variable
import pydicom

##
## Custom class for loading data
##
class MyCustomDataset(Dataset):
    def __init__(self, percent, direc, transform,args):
        self.data_root = direc
        self.args=args
        self.transform = transform
        if (os.path.exists((os.path.join(self.data_root, '/labels')))):
            self.names = np.array([name for name in os.listdir((os.path.join(self.data_root, '/labels'+name)))])
        else: self.names = np.array(sorted([name for name in os.listdir(self.data_root)]))
        if percent<0:self.names = [self.names[-1*percent-1]]  # treated as one index
        else: self.names = self.names[0:int(percent*len(self.names)//100)]
        self.count = len(self.names)
        #print(f'self count: {self.count}')
        #print('names:', self.names)

    def __getitem__(self, index):
        name = self.names[index]
        rayed = torch.Tensor()
        img = Image.fromarray(load_dicom_as_array((os.path.join(self.data_root, name))))
        img = self.transform(img)
        return (rayed, img, name)

    def __len__(self):
        return self.count

def load_dicom_as_array(filepath):
    dicom_file = pydicom.dcmread(filepath)
    image_array = dicom_file.pixel_array.astype(np.float32)    
    # Normalize the image if needed (e.g., to [0, 1] range)
    image_array -= np.min(image_array)
    image_array /= np.max(image_array)
    return image_array

def load_data(args):

    data_valid_loader, data_test_loader = [],[]


    data_valid = MyCustomDataset(100,args.data_path+'valid',transform=transforms.Compose([
                                                                                transforms.Resize((config.size, config.size)),
                                                                                transforms.ToTensor()
                                                                                ]),args=args)

    data_valid_loader = DataLoader(data_valid, batch_size=args.batch_size_valid, shuffle=False, num_workers=args.workers, drop_last=False)

    data_test = MyCustomDataset(-args.ntest,args.data_path+'test', transform=transforms.Compose([
                                                                                transforms.Resize((config.size, config.size)),
                                                                                transforms.ToTensor()
                                                                                ]),args=args)
    data_test_loader = DataLoader(data_test, batch_size=args.batch_size_valid, shuffle=False, num_workers=args.workers, drop_last=False)

    return (data_valid_loader, data_test_loader)


#
## Creating noisy version/scans after Radon from the truth
##
def create(truths, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if(config.angles != 0):rayed = config.fwd_op_mod(truths)
    else: rayed=truths.clone()
    rayed += Variable(config.noise * torch.randn(rayed.shape)).type_as(rayed)
    return rayed
