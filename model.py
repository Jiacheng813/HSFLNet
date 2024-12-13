import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50
from torch.nn import functional as F
from hypergraphs import HypergraphConv

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v
        #self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.visible.conv1(x)
        #x = self.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t
        #self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.thermal.conv1(x)
        #x = self.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class visible_moduleA(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_moduleA, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v
        #self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.visible.conv1(x)
        #x = self.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_moduleA(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_moduleA, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t
        #self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.thermal.conv1(x)
        #x = self.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0), nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)
        
        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        gate = self.sigmoid(W_y) # gating mechanism
        z = gate * W_y + (1 - gate) * x_h
        return z

class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(nn.Conv2d(self.low_dim//self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim//self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        gate = self.sigmoid(W_y) # gating mechanism
        z = gate * W_y + (1 - gate) * x_h
        return z

class MFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)
    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z


    def forward(self, x):
        out = self.wh(x)
        if self.affine:
            out = out * self.gamma + self.beta + x
        return out

class embed_net(nn.Module):
    def __init__(self, class_num, arch='resnet50',graphw=1.0, theta1=0.0, edge=256):
        super(embed_net, self).__init__()
        self.part_num = 4
        self.global_dim = 2048
        self.style_dim = 256
        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.thermal_moduleA = thermal_moduleA(arch=arch)
        self.visible_moduleA = visible_moduleA(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.MFA1 = MFA_block(256, 64, 0)
        self.MFA2 = MFA_block(512, 256, 1)
        
        self.l2norm = Normalize(2)

        self.bottleneck2 = nn.BatchNorm1d(self.style_dim)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.pattern_classifiers = nn.ModuleList([
            nn.Linear(self.style_dim, class_num, bias=False) for _ in range(self.part_num)
        ])
        for classifier in self.pattern_classifiers:
            classifier.apply(weights_init_classifier)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.activation = nn.Sigmoid()
        self.spatial_attention = nn.Conv2d(self.global_dim, self.part_num, kernel_size=1, stride=1, padding=0, bias=True)     
        self.pool_dim = self.global_dim + self.style_dim * self.part_num
        self.bottleneck = nn.BatchNorm1d(self.pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.pool_dim, class_num, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.weight_sep = 0.2

        self.hypergraph = HypergraphConv(
            in_features=self.style_dim,  # 输入通道数
            out_features=self.style_dim,  # 输出通道数
            features_height=24,  # 假设特征图高度为 24
            features_width=12,  # 假设特征图宽度为 12
            theta1=theta1,
            edges=edge
            )
        # self.graphw = graphw
        self.graphw = nn.Parameter(torch.tensor(graphw, requires_grad=True))
        self.conv_reduce = nn.Conv2d(in_channels=2048, out_channels=self.style_dim, kernel_size=1, stride=1, padding=0)
        self.bn_conv_reduce = nn.BatchNorm2d(self.style_dim)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x1_1, x1_2,x2_1,x2_2, modal=0):
        if modal == 0: #four-stream feature extracter
            x1_1 = self.visible_module(x1_1)
            x2_1 = self.thermal_module(x2_1)
            x1_2 = self.visible_moduleA(x1_2)
            x2_2 = self.thermal_moduleA(x2_2)
            x = torch.cat((x1_1, x1_2,x2_1,x2_2), 0)
        elif modal == 1: #In the test mode, merge original features with data-augmented features
            x1_1 = self.visible_module(x1_1)       
            x1_2 = self.visible_moduleA(x1_2)
            x_mix=(x1_1+x1_2)/2
            x=x_mix

            #x = torch.cat((x1_1, x1_1), 0)
            #x = torch.cat((x1_2, x1_2), 0)
        elif modal == 2:
            x2_1 = self.thermal_module(x2_1)
            x2_2 = self.thermal_moduleA(x2_2)
            x_mix=(x2_1+x2_2)/2
            x=x_mix

            #x = torch.cat((x2_1, x2_1), 0)
            #x = torch.cat((x2_2, x2_2), 0)

        # shared block
        x_low = x
        x = self.base_resnet.base.layer1(x) 
        x = self.MFA1(x, x_low)
        x_low = x
        x = self.base_resnet.base.layer2(x)
        x = self.MFA2(x, x_low)
        x = self.base_resnet.base.layer3(x)  
        x = self.base_resnet.base.layer4(x)
        global_feat = x
        b, c, h, w = x.shape

        #Generate part_num types of semantic style maps
        masks = self.spatial_attention(global_feat) #get style maps
        masks = self.activation(masks) # activation for maps，
        feats = []
        feat_logit_styles = []
        for i in range(self.part_num): #for each style map
            mask = masks[:, i:i+1, :, :]
            x = mask*global_feat
            x = self.conv_reduce(x) # use convolution to reduce the dimension
            x = self.bn_conv_reduce(x)  # add a BN

            # classification for each style with a standalone linear classifier
            x_pool = self.avgpool(x) # pooling
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
            feat = self.bottleneck2(x_pool) # BN
            feat_logit = self.pattern_classifiers[i](feat) 
            feat_logit_styles.append(feat_logit)
            
            # apply hypergraph convolution
            feat = self.graphw * self.hypergraph(x) 
            
            feat = F.avg_pool2d(feat, feat.size()[2:]) # pooling
            feat = feat.view(feat.size(0), -1)
            feats.append(feat)

        global_feat = F.avg_pool2d(global_feat, global_feat.size()[2:])
        global_feat = global_feat.view(global_feat.size(0), -1)
        feats.append(global_feat)

        # concatenate pooled style features with the pooled global feature
        feats = torch.cat(feats, 1)
        x_pool = feats            
  
        if self.training:
            masks = masks.view(b, self.part_num, w*h)
            loss_reg = torch.bmm(masks, masks.permute(0, 2, 1))
            loss_reg = torch.triu(loss_reg, diagonal = 1).sum() / (b * self.part_num * (self.part_num - 1) / 2)
            if loss_reg != 0 :
                loss_sep = loss_reg.float() * self.weight_sep
            else:
                loss_sep = 0
        
        feat = self.bottleneck(x_pool)
        feat_logit = self.classifier(feat)
        
        if self.training:         
            return x_pool, feat_logit, loss_sep, feat_logit_styles
        else:
            return self.l2norm(x_pool), self.l2norm(feat)
