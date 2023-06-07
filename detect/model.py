import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
from transformer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self, mode, SF_checkpoint):
        super(VGGBase, self).__init__()

        self.mode = mode
        self.SF_checkpoint = SF_checkpoint

        # Standard convolutional layers in VGG16
        #################################### RGB #####################################

        self.conv1_1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_1_1_bn = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv1_2_1_bn = nn.BatchNorm2d(64, affine=True)
        
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_1_1_bn = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2_2_1_bn = nn.BatchNorm2d(128, affine=True)
        
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_1_1_bn = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_2_1_bn = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3_1_bn = nn.BatchNorm2d(256, affine=True)

        self.pool3_1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  

        self.conv4_1_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_1_1_bn = nn.BatchNorm2d(512, affine=True)
        self.conv4_2_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_2_1_bn = nn.BatchNorm2d(512, affine=True)
        self.conv4_3_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3_1_bn = nn.BatchNorm2d(512, affine=True)

        #################################################################################

        #################################### Thermal ####################################

        self.conv1_1_2 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_1_2_bn = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv1_2_2_bn = nn.BatchNorm2d(64, affine=True)

        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_1_2_bn = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2_2_2_bn = nn.BatchNorm2d(128, affine=True)
        
        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_1_2_bn = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_2_2_bn = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3_2_bn = nn.BatchNorm2d(256, affine=True)

        self.pool3_2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  

        self.conv4_1_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_1_2_bn = nn.BatchNorm2d(512, affine=True)
        self.conv4_2_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_2_2_bn = nn.BatchNorm2d(512, affine=True)
        self.conv4_3_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3_2_bn = nn.BatchNorm2d(512, affine=True)

        if 'SF' in self.mode:
            self.conv1x1_sf_visible = nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=True)
            self.conv1x1_sf_visible.weight.data.normal_(0, 0.01)
            self.conv1x1_sf_visible.bias.data.fill_(0.01)

            self.conv1x1_sf_lwir = nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=True)
            self.conv1x1_sf_lwir.weight.data.normal_(0, 0.01)
            self.conv1x1_sf_lwir.bias.data.fill_(0.01)   
        else:
            raise ValueError
            sys.exit(1)

        # Load pretrained layers
        self.load_pretrained_layers()

    def forward(self, visible_image, lwir_image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """

        #################################### RGB #####################################
        out_1 = F.relu(self.conv1_1_1_bn(self.conv1_1_1(visible_image)))
        out_1 = F.relu(self.conv1_2_1_bn(self.conv1_2_1(out_1)))
        
        if 'AF' in self.mode:
            conv1_2_feats_visible = out_1

        out_1 = self.pool1_1(out_1)

        ################################### Thermal ##################################
        
        out_2 = F.relu(self.conv1_1_2_bn(self.conv1_1_2(lwir_image)))
        out_2 = F.relu(self.conv1_2_2_bn(self.conv1_2_2(out_2)))

        if 'AF' in self.mode:
            conv1_2_feats_lwir = out_2
        
        out_2 = self.pool1_2(out_2)

        #################################### RGB #####################################

        out_1 = F.relu(self.conv2_1_1_bn(self.conv2_1_1(out_1)))
        out_1 = F.relu(self.conv2_2_1_bn(self.conv2_2_1(out_1)))
        out_1 = self.pool2_1(out_1)

        out_1 = F.relu(self.conv3_1_1_bn(self.conv3_1_1(out_1)))
        out_1 = F.relu(self.conv3_2_1_bn(self.conv3_2_1(out_1)))
        out_1 = F.relu(self.conv3_3_1_bn(self.conv3_3_1(out_1)))
        
        if 'SF' in self.mode:
            out_1 = F.relu(self.conv1x1_sf_visible(out_1))
        else:
            raise ValueError
            sys.exit(1)

        out_1 = self.pool3_1(out_1)
        
        ################################### Thermal ##################################

        out_2 = F.relu(self.conv2_1_2_bn(self.conv2_1_2(out_2)))
        out_2 = F.relu(self.conv2_2_2_bn(self.conv2_2_2(out_2)))  
        out_2 = self.pool2_2(out_2)

        out_2 = F.relu(self.conv3_1_2_bn(self.conv3_1_2(out_2)))
        out_2 = F.relu(self.conv3_2_2_bn(self.conv3_2_2(out_2)))
        out_2 = F.relu(self.conv3_3_2_bn(self.conv3_3_2(out_2)))
        
        if 'SF' in self.mode:
            out_2 = F.relu(self.conv1x1_sf_lwir(out_2))
        else:
            raise ValueError
            sys.exit(1)  

        out_2 = self.pool3_1(out_2)   

        #################################### RGB #####################################
        
        out_1 = F.relu(self.conv4_1_1_bn(self.conv4_1_1(out_1)))    
        out_1 = F.relu(self.conv4_2_1_bn(self.conv4_2_1(out_1)))    
        out_1 = F.relu(self.conv4_3_1_bn(self.conv4_3_1(out_1)))

        conv4_3_feats_visible = out_1

        ################################### Thermal #####################################

        out_2 = F.relu(self.conv4_1_2_bn(self.conv4_1_2(out_2)))    
        out_2 = F.relu(self.conv4_2_2_bn(self.conv4_2_2(out_2)))    
        out_2 = F.relu(self.conv4_3_2_bn(self.conv4_3_2(out_2)))    

        conv4_3_feats_lwir = out_2

        #################################################################################

        return conv4_3_feats_visible, conv4_3_feats_lwir

    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        state_dict = self.state_dict()

        # If need, Load vgg16_bn
        param_names = list(state_dict.keys())
        for name in param_names[:]:
            if 'bn' in name:
                param_names.remove(name)

        param_names = param_names[:-4]

        middle_index = len(param_names) // 2

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        # Transfer until conv4_3 parameters from pretrained model to current model
        for i, param in enumerate(param_names[:middle_index]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Transfer until conv4_3 parameters from pretrained model to current model
        for i, param in enumerate(param_names[middle_index:]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)

        print("INFO: Load base model")
        

class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()

        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, out):
        return self.relu(out + 3) / 6


class HSwitch(nn.Module):
    def __init__(self, inplace=True):
        super(HSwitch, self).__init__()

        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, out):
        return out * self.sigmoid(out)


class CoordinateAttention(nn.Module):
    """
    Coorinate Attention Module
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.coord_conv = nn.Sequential(nn.Conv2d(in_channels=inp, out_channels=mip, kernel_size=1), 
                                        nn.BatchNorm2d(mip), 
                                        HSwitch()
                                        )

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

        print('INFO: Load Coordinate Attention Module')

    def forward(self, out):
        identity = out

        _, c, h, w = out.size()
        out_h = self.pool_h(out)
        out_w = self.pool_w(out).permute(0, 1, 3, 2)

        out = torch.cat([out_h, out_w], dim=2)
        out = self.coord_conv(out)

        out_h, out_w = torch.split(out, [h, w], dim=2)
        out_w = out_w.permute(0, 1, 3, 2)

        out_h = self.conv_h(out_h).sigmoid()
        out_w = self.conv_w(out_w).sigmoid()

        out = identity * out_w * out_h

        return out


class Transformer_Fusion(nn.Module):
    """
    Fusion convolutions to combine multimodal inputs
    before conv4_3 to adopt attention module, CBAM
    """
    def __init__(self, mode, SF_checkpoint):
        super(Transformer_Fusion, self).__init__()

        self.mode = mode
        self.SF_checkpoint = SF_checkpoint

        # Fusion
        self.NIN = nn.Conv2d(512 * 3, 512, kernel_size=1)

        self.transformer = FeatureTransformer(num_layers=2, d_model=512, nhead=1, attention_type="swin", ffn_dim_expansion=4)
        self.init_NIN()

        print("INFO: Load Transformer")
        print("INFO: Load H-Fusion model")

    def forward(self, conv4_3_feats_visible, conv4_3_feats_lwir, d_prob):
        # Transformer
        out_1, out_2 = self.transformer(conv4_3_feats_visible, conv4_3_feats_lwir, attn_num_splits=2)
        
        _, c, h, w = conv4_3_feats_visible.shape

        d_prob = torch.FloatTensor(d_prob)
        d_prob = d_prob.unsqueeze(1).unsqueeze(2).expand(-1, c, h, w).to(device)
  
        # dprob * visible_feats + (1-dprob) * lwir_feats
        dn_feats = torch.add(torch.mul(d_prob, conv4_3_feats_visible), torch.mul((1 - d_prob), conv4_3_feats_lwir))

        # Fusion
        out = torch.cat((out_1, out_2, dn_feats), dim=1)   
        out = self.NIN(out)
        
        conv4_3_feats = out

        # Lower-level feature maps
        return conv4_3_feats

    def init_NIN(self):
        """
        Initialize 1x1 convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
                

class ModifiedSSD(nn.Module):
    """
    Convolutions to combine after attention
    """  
    def __init__(self):
        super(ModifiedSSD, self).__init__()
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512, affine=True)

        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Replacements for FC6 and FC7 in VGG16
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv6_bn = nn.BatchNorm2d(1024)

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv7_bn = nn.BatchNorm2d(1024)

        self.load_pretrained_layers() 

    def forward(self, conv4_3_feats):
        # Fusion
        out = self.pool4(conv4_3_feats)  # (N, 512, 19, 19)

        out = F.relu(self.conv5_1_bn(self.conv5_1(out)))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_2_bn(self.conv5_2(out)))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_3_bn(self.conv5_3(out)))  # (N, 512, 19, 19)
        out = self.pool5(out)

        out = F.relu(self.conv6_bn(self.conv6(out)))  # (N, 1024, 19, 19)
        out = F.relu(self.conv7_bn(self.conv7(out)))  # (N, 1024, 19, 19)

        conv7_feats = out

        # Mid-level feature maps
        return conv7_feats

    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        state_dict = self.state_dict()

        param_names = list(state_dict.keys())
        for name in param_names[:]:
            if 'bn' in name:
                param_names.remove(name)

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        pretrained_param_names = pretrained_param_names[20:-6]

        for i, param in enumerate(param_names[:6]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 25088) > (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096) > (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding

        self.load_state_dict(state_dict)


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()
        
        # Auxiliary/additional convolutions on top of the VGG base
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)   # stride = 1, by default
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1
        
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)   # dim. reduction because padding = 0

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)   # dim. reduction because padding = 0

        # Initialize convolutions' parameters as Xaivers init
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """        
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)
        out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
        conv8_2_feats = out  # (N, 512, 10, 10)

        out = F.relu(self.conv9_1(out))  # (N, 128, 10, 10)
        out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
        conv9_2_feats = out  # (N, 256, 5, 5)

        out = F.relu(self.conv10_1(out))  # (N, 128, 5, 5)
        out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)
        conv10_2_feats = out  # (N, 256, 3, 3)

        out = F.relu(self.conv11_1(out))  # (N, 128, 3, 3)
        conv11_2_feats = F.relu(self.conv11_2(out))  # (N, 256, 1, 1)

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        """
        Forward propagation.

        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 10, 10)
        :param conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, 5, 5)
        :param conv10_2_feats: conv10_2 feature map, a tensor of dimensions (N, 256, 3, 3)
        :param conv11_2_feats: conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 4 * 4), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        # In other words, reshaping in pyTorch is only possible if the original tensor is stored in a contiguous chunk of memory.
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                   self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        # A total of 8732 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],
                                   dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores


class SSD300(nn.Module):
    """
    The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, n_classes, mode=None, SF_checkpoint=None):
        super(SSD300, self).__init__()

        if mode == None :
            print("Mode Error")
            sys.exit(1)

        self.n_classes = n_classes
        self.mode = mode
        self.SF_checkpoint = SF_checkpoint

        self.base = VGGBase(self.mode, self.SF_checkpoint)
        self.transformer = Transformer_Fusion(self.mode, self.SF_checkpoint)
        self.attention = CoordinateAttention(inp=512, oup=512)
        self.modified = ModifiedSSD()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, visible_image, lwir_image, d_prob):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats_visible, conv4_3_feats_lwir = self.base(visible_image, lwir_image)
        conv4_3_feats = self.transformer(conv4_3_feats_visible, conv4_3_feats_lwir, d_prob)
        conv4_3_feats = self.attention(conv4_3_feats)

        # Rescale conv4_3 after L2 norm
        # As ParseNet say, Feature map dim of conv4_3(38*38) is higher than others, so L2Normalization layer added
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # Euclidean, (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        conv7_feats = self.modified(conv4_3_feats)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,
                                               conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores, conv11_2_feats
    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {'conv4_3' : 38,
                     'conv7'   : 19,
                     'conv8_2' : 10,
                     'conv9_2' : 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3' : 0.1,
                      'conv7'   : 0.2,
                      'conv8_2' : 0.375,
                      'conv9_2' : 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3' : [1., 2., 0.5],
                         'conv7'   : [1., 2., 3., 0.5, .333],
                         'conv8_2' : [1., 2., 3., 0.5, .333],
                         'conv9_2' : [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        # prior_boxes = [[cx, cy, w, h], ...]
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0) # (1)
        n_priors = self.priors_cxcy.size(0) # (8732, 4) > (8732)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score)).bool().to(device)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = suppress | (overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[~suppress])
                image_labels.append(torch.LongTensor((~suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[~suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        
        batch_size = predicted_locs.size(0) # (8, 8732, 4) > (8)
        n_priors = self.priors_cxcy.size(0) # (8732, 4) > (8732)
        n_classes = predicted_scores.size(2)    # (8, 8732, 21) > (21)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance
        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss
