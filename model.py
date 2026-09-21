from copy import deepcopy
import logging
from torch import nn, load, flatten
import torch.nn.functional as F
from torchvision import models
#from torchvision.models.feature_extraction import create_feature_extractor

#================================================
class Generic():
    name = None
    num_classes:int = 0
    fc_name = ""
    
    def fullnet(self, custom_weights = None, freeze_bn = True): 
        # custom, non default, weights expect already datasize num of out classes 
        # and other netowrk changes to be applyed before weights loading
        if custom_weights is not None: self.preload(custom_weights, freeze_bn)  
        self._fc = self.model.fc # save (possiblly) trained fc
        logging.info(f"Using {self.name} {'pretrained' if custom_weights is True else ''} full model ")
        return self.model
    
    # just turns the init() model into one without fc, after saving it
    def backbone(self, custom_weights = None, freeze_bn = True):      
        if custom_weights is not None: self.preload(custom_weights, freeze_bn)                              
        self._fc = self.model.fc # save (possiblly) trained fc            
        #net = nn.Sequential(*list(net.children())[:-1])  # remove fc
        #net = create_feature_extractor(net, return_nodes={"avgpool": "features"})
        self.model.fc = nn.Identity(); # type: ignore
        logging.info(f"Using {self.backbone_name} backbone... ")             
        return self.model

    def fc(self, empty=False):  
        # returns SAVED (possibly trained) model fc, NOT THE CURRENT self.model.fc !
        logging.info(f"Using {Generic.fc_name} {'new empty' if empty else 'pretrained copy of' } Fully Connected Classifier ")
        if empty: self._fc = nn.Linear(self.in_features, self.num_classes)
        return self._fc  #None if has never been pretrained and saved: should never be None with empty=False
      
    def __init__(self, model_name, out:int, pretrain = None, freeze_bn = True): 
        # protocol is: DEFAULT weights are applicable here at init time
        # other custom weights must be applied in instance method calls
        Generic.name = self.name = model_name  #get actual type name from subtype
        self.model = getattr(models, self.name)(weights = None)  # models.resnet18(weights = None) 
        self.in_features = self.model.fc.in_features
        if pretrain is not None: self.preload(weights = pretrain, freeze = freeze_bn)                              
        Generic.num_classes = self.num_classes = out            
        Generic.fc_name = f"LINEAR[{self.in_features}, {self.num_classes}]"
        self.model.fc = nn.Linear(self.in_features, self.num_classes) #reshape fc layer before training
        self.backbone_name = f"{self.name}[BACK]"
        logging.info(f"Istantiated {self.name} model... ")             
                       
    def freezebn(self):    
        logging.info(f"freezing bach norm layers stats") 
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.requires_grad_(False)
    
    def preload(self, weights, freeze = True):
        #custom_weights.get_state_dict(progress=True, check_hash=True)  # to get it ftom url
        w = load(weights, weights_only=False, map_location='mps')   
        self.model.load_state_dict(w)
        if freeze: self.freezebn() # Keep BatchNorm statistics from ImageNet; only affine params are trained
        logging.info(f"{self.name} preloaded with weights from {weights}")    

     # def forward(self, x):
    #     if back: 
    #       x = self.back(x)
    #       x = x.squeeze() # torch.flatten(x, 1)
    #     if head: x = self.head(x)
    #     return x

#================================================
#========================RESNET50 version        
class Resnet50(Generic):  
    def __init__(self, out): 
        #defaults to a pure resnet50 with no ajdustment at all        
        super().__init__("resnet50", out) # models.resnet18(weights = None) 
 
#========================RESNET18 IMPLEMENTATION
class Resnet18(Generic):  
    def __init__(self, out, pretrain = None, freeze = True): 
        #defaults to a pure resnet18 with no ajdustment at all        
        super().__init__("resnet18", out, pretrain, freeze) # models.resnet18(weights = None) 
        
#========================RESNET18 version adapted for 32x32 image datasets (like CIFAR)     
class Resnet18tr(Resnet18): 
    def adapt2CIFAR(self):
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.model.maxpool = nn.Identity() # type: ignore
        logging.info(f"- {self.backbone_name} modified to suit 32x32 images")
    
    def backbone(self, custom_weights = None, freeze_bn = True): 
        self.adapt2CIFAR()        
        return super().backbone(custom_weights, freeze_bn)     
    
    def fullnet(self, custom_weights = None, freeze_bn = True): 
        self.adapt2CIFAR()
        return super().fullnet(custom_weights, freeze_bn)
    
    def __init__(self, out, pretrain = None, freeze = True):
        super().__init__(out, pretrain, freeze)
              
   
#========================RESNET Custom version adapted FOR CIFAR      
class ResNetCustom(nn.Module):
    #from torchvision.models.resnet import conv3x3
    def __init__(self, block, layers, num_classes=100):
        super(ResNetCustom, self).__init__()
        self.in_channels = 32
        
        # Initial convolution layer with BatchNorm, ReLU, and Dropout
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        
        # Residual blocks
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        
        # Max Pooling and Output layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def conv3x3(self, in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
        """3x3 convolution with padding"""
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                self.conv3x3(self.in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = F.max_pool2d(x, kernel_size=4)
        x = flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)  
        
        return x