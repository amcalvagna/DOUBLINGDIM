from sklearn.model_selection import train_test_split
from torchvision import  datasets, transforms
#from utils import convert_to_rgb
from torch.utils.data import Dataset #, ConcatDataset, TensorDataset
import numpy as np

# WM38 38-class mapping
classes_dict = {
    "00000000": 0, "10000000": 1, "01000000": 2, "00100000": 3,
    "00010000": 4, "00001000": 5, "00000100": 6, "00000010": 7,
    "00000001": 8, "10100000": 9, "10010000": 10, "10001000": 11,
    "10000010": 12, "01100000": 13, "01010000": 14, "01001000": 15,
    "01000010": 16, "00101000": 17, "00100010": 18, "00011000": 19,
    "00010010": 20, "00001010": 21, "10101000": 22, "10100010": 23,
    "10011000": 24, "10010010": 25, "10001010": 26, "01101000": 27,
    "01100010": 28, "01011000": 29, "01010010": 30, "01001010": 31,
    "00101010": 32, "00011010": 33, "10101010": 34, "10011010": 35,
    "01101010": 36, "01011010": 37
}

# WM38 8-class mapping for second dataset
failure_mapping = {
    'Center':    [1,0,0,0,0,0,0,0], 'Donut': [0,1,0,0,0,0,0,0],
    'Edge-Loc':  [0,0,1,0,0,0,0,0], 'Edge-Ring': [0,0,0,1,0,0,0,0],
    'Loc':       [0,0,0,0,1,0,0,0], 'Near-full': [0,0,0,0,0,1,0,0],
    'Scratch':   [0,0,0,0,0,0,1,0], 'Random': [0,0,0,0,0,0,0,1],
    'none':      [0,0,0,0,0,0,0,0]
}

# WM38 one hot encoding
strict_mapping = {
    "00000000": 0, "10000000": 1, "01000000": 2, "00100000": 3,
    "00010000": 4, "00001000": 5, "00000100": 6, "00000010": 7,
    "00000001": 8,
}

# for Wafer Dataset 38k data conversion 
def convert_to_rgb(image):
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    h, w = image.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[image == 0] = [0, 0, 0]
    rgb[image == 1] = [0, 255, 0] 
    rgb[image == 2] = [255, 0, 0]
    return rgb

class MergedDataset(Dataset):
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2
        self.len1 = len(ds1)

    def __len__(self):
        return self.len1 + len(self.ds2)

    def __getitem__(self, idx):
        if idx < self.len1:
            return self.ds1[idx]
        return self.ds2[idx - self.len1]

# class CifarBase():
#     @classmethod
#     def combine(cls, dataset1, dataset2):
#         # train_data, train_labels = torch.tensor(dataset1.data,dtype=torch.float32), torch.tensor(dataset1.targets,dtype=torch.long)
#         # test_data, test_labels = torch.tensor(dataset2.data,dtype=torch.float32), torch.tensor(dataset2.targets,dtype=torch.long)
#         # # Concatenate train and test data and targets
#         # all_data = torch.cat((train_data, test_data), dim=0).permute(0,3,1,2) 
#         # all_targets = torch.cat((train_labels, test_labels), dim=0)
#         # # Create a new TensorDataset combining all samples
#         # combined_dataset = TensorDataset(all_data, all_targets)
#         combined_dataset = ConcatDataset([train, test])
#         return combined_dataset

# class CifarDataset(Dataset):   
#     def __init__(self, images, labels, transform=None):
#         self.images = images
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         img = self.images[idx]
#         if self.transform:
#             img = self.transform(img)
#         return img, self.labels[idx]

class CoreDataset(Dataset): 
    def __init__(self, dataset, coreset) -> None:
       self.dataset = dataset
       self.coreset = coreset
       
    def __len__(self): 
        return len(self.coreset)
    
    def __getitem__(self, index):
        return self.dataset.__getitem__(self.coreset[index])

class WaferDataset38(Dataset):   
    def __init__(self, images, labels, transform=None):
        self.data = images
        self.targets = [
            classes_dict["".join(str(int(x)) for x in one_hot)]
            for one_hot in labels
        ]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        # if len(img.shape) == 3 and img.shape[0] == 1:
        #     img = img.squeeze(0)
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]




#================================

class CIFAR10():
   
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD  = [0.2470, 0.2435, 0.2616] #[0.2023, 0.1994, 0.2010]

    name = "CIFAR10" 
    path = f"./dataset/{name}"   
    num_classes = 10
    shape = (3, 32, 32)

    transform = transforms.Compose([
        #transforms.Resize((224,224)), 4resnet18
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    train, test = None, None
    
    @classmethod
    def dataset(cls):     
           # get dataset only once
        cls.train = datasets.CIFAR10(root=cls.path, train=True, download=False, transform=cls.transform) if cls.train is None else cls.train
        cls.test = datasets.CIFAR10(root=cls.path, train=False, download=False, transform=cls.transform) if cls.test is None else cls.test
        return cls.train, cls.test
    #n_samples = len(train_dataset().targets)
 
    @classmethod
    def combined(cls):
        #return ConcatDataset(cls.dataset())
        train, test = cls.dataset()
        #return MergedDataset(train, test)
        train.data = np.concatenate([train.data, test.data])
        train.targets += test.targets
        return train
    
class CIFAR100():
   
    CIFAR100_MEAN= [0.5071, 0.4865, 0.4409]
    CIFAR100_STD= [0.2673, 0.2564, 0.2762]

    name = "CIFAR100" 
    path = f"./dataset/{name}"   
    num_classes = 100
    shape = (3, 32, 32)

    transform = transforms.Compose([
        #transforms.Resize((224,224)), 4resnet18
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

    train, test = None, None

    @classmethod
    def dataset(cls):     
           # get dataset only once
        cls.train = datasets.CIFAR100(root=cls.path, train=True, download=True, transform=cls.transform) if cls.train is None else cls.train
        cls.test = datasets.CIFAR100(root=cls.path, train=False, download=True, transform=cls.transform) if cls.test is None else cls.test
        return cls.train, cls.test
    #n_samples = len(train_dataset().targets)

    @classmethod
    def combined(cls):
        #return ConcatDataset(cls.dataset())
        train, test = cls.dataset()
        #return MergedDataset(train, test)
        train.data = np.concatenate([train.data, test.data])
        train.targets += test.targets
        return train

class Cifar100RN(CIFAR100):
    # ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    name = "CIFAR100RN" 
    path = f"./dataset/CIFAR100"   
    num_classes = 100
    shape = (3, 224, 224)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        # Strong but standard policy; works well with CIFAR-like data
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train, test = None, None
    
    @classmethod
    def dataset(cls):     
           # get dataset only once
        cls.train = datasets.CIFAR100(root=cls.path, train=True, download=True, transform=cls.train_transform) if cls.train is None else cls.train
        cls.test = datasets.CIFAR100(root=cls.path, train=False, download=True, transform=cls.test_transform) if cls.test is None else cls.test
        return cls.train, cls.test
    #n_samples = len(train_dataset().targets)
    
    # @classmethod
    # def combined(cls):
    #     #return ConcatDataset(cls.dataset())
    #     train, test = cls.dataset()
    #     #return MergedDataset(train, test)
    #     train.data = np.concatenate([train.data, test.data])
    #     train.targets += test.targets
    #     return train


class WM38():  
    # Download latest version
    #WM38k_path = kagglehub.dataset_download("co1d7era/mixedtype-wafer-defect-datasets")
    name = "WM38" 
    path = f"./dataset/{name}/Wafer_Map_Datasets.npz"   
    num_classes = 38
    shape = (3, 52, 52)

    rgb_transform = transforms.Compose([
        #transforms.Lambda(lambda x: x.squeeze() if len(x.shape) == 3 and x.shape[0] == 1 else x),
        transforms.Lambda(convert_to_rgb),
        #transforms.ToPILImage(), #unrequired
        #transforms.Resize((224, 224)), #unrequired
        transforms.ToTensor(), #dtype convert to float32 and scale to 0-1
    ])

    data38 = np.load(path)
    imgs38, labs38 = data38["arr_0"], data38["arr_1"]
    i38_tr, i38_te, l38_tr, l38_te = train_test_split(imgs38, labs38, test_size=0.1, random_state=42)
    train, test = None, None

    @classmethod
    def combined(cls):
        # Create a new TensorDataset combining all samples
        combined_dataset = WaferDataset38(cls.imgs38, cls.labs38, cls.rgb_transform)
        return combined_dataset
    
    @classmethod
    def dataset(cls):     
        # get dataset only once
        cls.train = WaferDataset38(cls.i38_tr, cls.l38_tr, cls.rgb_transform) 
        cls.test = WaferDataset38(cls.i38_te, cls.l38_te, cls.rgb_transform) 

        return cls.train, cls.test
        #n_samples = len(train_dataset().targets)
