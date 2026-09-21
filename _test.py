from matplotlib.lines import lineStyles
from pandas import Int32Dtype
import torch
from torchvision.models import resnet18

# A = [[1,2],[3,4]]
# B = A 
# B[0][0] = 99


# if A[0][0] == 99 :
#     print("Memory test : shared object")
# else :
#     print("Memory test : cloned objects")


# A = None  # free memory

# if B is not None :
#     print("Memory test : shared object still accessible")
# else :
#     print("Memory test : shared object removed")


# model = resnet18(weights = None)
# # Modify the first conv layer (CIFAR-10 has 32x32 images, not 224x224)
# model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# model.maxpool = torch.nn.Identity()  # type: ignore # remove the 7x7 -> 3x3 reduction
# print(model.fc.in_features)
# print(model.name)
# model.fc = torch.nn.Linear(model.fc.in_features, 10)
    
# A =["conv1.weight", "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", "layer1.0.conv1.weight", "layer1.0.bn1.weight", "layer1.0.bn1.bias", "layer1.0.bn1.running_mean", "layer1.0.bn1.running_var", "layer1.0.conv2.weight", "layer1.0.bn2.weight", "layer1.0.bn2.bias", "layer1.0.bn2.running_mean", "layer1.0.bn2.running_var", "layer1.1.conv1.weight", "layer1.1.bn1.weight", "layer1.1.bn1.bias", "layer1.1.bn1.running_mean", "layer1.1.bn1.running_var", "layer1.1.conv2.weight", "layer1.1.bn2.weight", "layer1.1.bn2.bias", "layer1.1.bn2.running_mean", "layer1.1.bn2.running_var", "layer2.0.conv1.weight", "layer2.0.bn1.weight", "layer2.0.bn1.bias", "layer2.0.bn1.running_mean", "layer2.0.bn1.running_var", "layer2.0.conv2.weight", "layer2.0.bn2.weight", "layer2.0.bn2.bias", "layer2.0.bn2.running_mean", "layer2.0.bn2.running_var", "layer2.0.downsample.0.weight", "layer2.0.downsample.1.weight", "layer2.0.downsample.1.bias", "layer2.0.downsample.1.running_mean", "layer2.0.downsample.1.running_var", "layer2.1.conv1.weight", "layer2.1.bn1.weight", "layer2.1.bn1.bias", "layer2.1.bn1.running_mean", "layer2.1.bn1.running_var", "layer2.1.conv2.weight", "layer2.1.bn2.weight", "layer2.1.bn2.bias", "layer2.1.bn2.running_mean", "layer2.1.bn2.running_var", "layer3.0.conv1.weight", "layer3.0.bn1.weight", "layer3.0.bn1.bias", "layer3.0.bn1.running_mean", "layer3.0.bn1.running_var", "layer3.0.conv2.weight", "layer3.0.bn2.weight", "layer3.0.bn2.bias", "layer3.0.bn2.running_mean", "layer3.0.bn2.running_var", "layer3.0.downsample.0.weight", "layer3.0.downsample.1.weight", "layer3.0.downsample.1.bias", "layer3.0.downsample.1.running_mean", "layer3.0.downsample.1.running_var", "layer3.1.conv1.weight", "layer3.1.bn1.weight", "layer3.1.bn1.bias", "layer3.1.bn1.running_mean", "layer3.1.bn1.running_var", "layer3.1.conv2.weight", "layer3.1.bn2.weight", "layer3.1.bn2.bias", "layer3.1.bn2.running_mean", "layer3.1.bn2.running_var", "layer4.0.conv1.weight", "layer4.0.bn1.weight", "layer4.0.bn1.bias", "layer4.0.bn1.running_mean", "layer4.0.bn1.running_var", "layer4.0.conv2.weight", "layer4.0.bn2.weight", "layer4.0.bn2.bias", "layer4.0.bn2.running_mean", "layer4.0.bn2.running_var", "layer4.0.downsample.0.weight", "layer4.0.downsample.1.weight", "layer4.0.downsample.1.bias", "layer4.0.downsample.1.running_mean", "layer4.0.downsample.1.running_var", "layer4.1.conv1.weight", "layer4.1.bn1.weight", "layer4.1.bn1.bias", "layer4.1.bn1.running_mean", "layer4.1.bn1.running_var", "layer4.1.conv2.weight", "layer4.1.bn2.weight", "layer4.1.bn2.bias", "layer4.1.bn2.running_mean", "layer4.1.bn2.running_var", "fc.weight", "fc.bias"]
# B = ["backbone.0.weight", "backbone.1.weight", "backbone.1.bias", "backbone.1.running_mean", "backbone.1.running_var", "backbone.1.num_batches_tracked", "backbone.4.0.conv1.weight", "backbone.4.0.bn1.weight", "backbone.4.0.bn1.bias", "backbone.4.0.bn1.running_mean", "backbone.4.0.bn1.running_var", "backbone.4.0.bn1.num_batches_tracked", "backbone.4.0.conv2.weight", "backbone.4.0.bn2.weight", "backbone.4.0.bn2.bias", "backbone.4.0.bn2.running_mean", "backbone.4.0.bn2.running_var", "backbone.4.0.bn2.num_batches_tracked", "backbone.4.1.conv1.weight", "backbone.4.1.bn1.weight", "backbone.4.1.bn1.bias", "backbone.4.1.bn1.running_mean", "backbone.4.1.bn1.running_var", "backbone.4.1.bn1.num_batches_tracked", "backbone.4.1.conv2.weight", "backbone.4.1.bn2.weight", "backbone.4.1.bn2.bias", "backbone.4.1.bn2.running_mean", "backbone.4.1.bn2.running_var", "backbone.4.1.bn2.num_batches_tracked", "backbone.5.0.conv1.weight", "backbone.5.0.bn1.weight", "backbone.5.0.bn1.bias", "backbone.5.0.bn1.running_mean", "backbone.5.0.bn1.running_var", "backbone.5.0.bn1.num_batches_tracked", "backbone.5.0.conv2.weight", "backbone.5.0.bn2.weight", "backbone.5.0.bn2.bias", "backbone.5.0.bn2.running_mean", "backbone.5.0.bn2.running_var", "backbone.5.0.bn2.num_batches_tracked", "backbone.5.0.downsample.0.weight", "backbone.5.0.downsample.1.weight", "backbone.5.0.downsample.1.bias", "backbone.5.0.downsample.1.running_mean", "backbone.5.0.downsample.1.running_var", "backbone.5.0.downsample.1.num_batches_tracked", "backbone.5.1.conv1.weight", "backbone.5.1.bn1.weight", "backbone.5.1.bn1.bias", "backbone.5.1.bn1.running_mean", "backbone.5.1.bn1.running_var", "backbone.5.1.bn1.num_batches_tracked", "backbone.5.1.conv2.weight", "backbone.5.1.bn2.weight", "backbone.5.1.bn2.bias", "backbone.5.1.bn2.running_mean", "backbone.5.1.bn2.running_var", "backbone.5.1.bn2.num_batches_tracked", "backbone.6.0.conv1.weight", "backbone.6.0.bn1.weight", "backbone.6.0.bn1.bias", "backbone.6.0.bn1.running_mean", "backbone.6.0.bn1.running_var", "backbone.6.0.bn1.num_batches_tracked", "backbone.6.0.conv2.weight", "backbone.6.0.bn2.weight", "backbone.6.0.bn2.bias", "backbone.6.0.bn2.running_mean", "backbone.6.0.bn2.running_var", "backbone.6.0.bn2.num_batches_tracked", "backbone.6.0.downsample.0.weight", "backbone.6.0.downsample.1.weight", "backbone.6.0.downsample.1.bias", "backbone.6.0.downsample.1.running_mean", "backbone.6.0.downsample.1.running_var", "backbone.6.0.downsample.1.num_batches_tracked", "backbone.6.1.conv1.weight", "backbone.6.1.bn1.weight", "backbone.6.1.bn1.bias", "backbone.6.1.bn1.running_mean", "backbone.6.1.bn1.running_var", "backbone.6.1.bn1.num_batches_tracked", "backbone.6.1.conv2.weight", "backbone.6.1.bn2.weight", "backbone.6.1.bn2.bias", "backbone.6.1.bn2.running_mean", "backbone.6.1.bn2.running_var", "backbone.6.1.bn2.num_batches_tracked", "backbone.7.0.conv1.weight", "backbone.7.0.bn1.weight", "backbone.7.0.bn1.bias", "backbone.7.0.bn1.running_mean", "backbone.7.0.bn1.running_var", "backbone.7.0.bn1.num_batches_tracked", "backbone.7.0.conv2.weight", "backbone.7.0.bn2.weight", "backbone.7.0.bn2.bias", "backbone.7.0.bn2.running_mean", "backbone.7.0.bn2.running_var", "backbone.7.0.bn2.num_batches_tracked", "backbone.7.0.downsample.0.weight", "backbone.7.0.downsample.1.weight", "backbone.7.0.downsample.1.bias", "backbone.7.0.downsample.1.running_mean", "backbone.7.0.downsample.1.running_var", "backbone.7.0.downsample.1.num_batches_tracked", "backbone.7.1.conv1.weight", "backbone.7.1.bn1.weight", "backbone.7.1.bn1.bias", "backbone.7.1.bn1.running_mean", "backbone.7.1.bn1.running_var", "backbone.7.1.bn1.num_batches_tracked", "backbone.7.1.conv2.weight", "backbone.7.1.bn2.weight", "backbone.7.1.bn2.bias", "backbone.7.1.bn2.running_mean", "backbone.7.1.bn2.running_var", "backbone.7.1.bn2.num_batches_tracked", "head.fc.weight", "head.fc.bias"]

# print("Parameters mapping test :")
# for a,b in zip(A,B):
#     print(f"{a}  -->  {b}") 

# from utils import *
# _,_, test_dataset = ds.dataset() 
# model = nnet.fullnet(pretrained_weights, ds.num_classes) # reset all and then load 
# test(nnet, test_dataset)

# from utils import *
# clean_percentage_keys("./dd/save/_LINEAR[512,10]_CIFAR10_coresets_100.plots", "./dd/save/LINEAR[512,10]_CIFAR10_coresets_100.plots")


# train =  TensorDataset(data['train']["feats"], data['train']["labels"])
# train, val = split(np.zeros(len(train.targets)), train.targets)):


# def counter():
#     counter.value += 1
#     return counter.value

# counter.value = 0

# print(counter())  # 1
# print(counter())  # 2
# print(counter())  # 3

# import timm

# model = timm.create_model("resnet18_cifar100", pretrained=True)

#a = torch.rand(4, 4)
a = torch.randint(0, 10, (4, 4))
#b = torch.rand(1, 4)
b = torch.randint(0, 10, (1, 4))

#y = torch.abs(a - b).amax(1)
#y = a.min(dim=0)
mask = a[:,0] <= 3
y = torch.nonzero(mask, as_tuple=True)[0]
y = a[y,:]


print(a)  # torch.Size([4, 4])
print(b)  # torch.Size([1, 4])
print(y)  # torch.Size([4, 4])