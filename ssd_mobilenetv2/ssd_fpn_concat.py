'''
code by zzg 2020-11-16
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
import mobilenetv2 as mobilenetv2
from data.config import USE_SE
from attention import Bottleneck, SEModule, ECAModule, Upsample, DWConv2d

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        self.cfg = voc     #(coco, voc)[num_classes == 4]

        self.priorbox = PriorBox(self.cfg)

        #self.priors = Variable(self.priorbox.forward(), volatile=True)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
            #print(self.priors.size())

        self.size = size

        # SSD network
        #self.vgg = nn.ModuleList(base)
        self.mobilenet = base

        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            # self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            self.detect = Detect()
        
        self.SB_256 = SB(256)
        self.SB_640 = SB(640)
        self.SB_960 = SB(960)

        # =====bobo新增==================
        # layer19_2到layer19_1
        self.upsample_256_256 = Upsample(5)
        self.conv_256_128 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)

        ##add后512-->1280 layer19_1到layer19 上采样，尺度大一倍
        self.upsample_640_640 = Upsample(10)
        self.conv_640_320 = nn.Conv2d(in_channels=640, out_channels=320, kernel_size=1, stride=1)

        # layer19到layer19  尺度不变
        self.conv_1280_640 = nn.Conv2d(in_channels=1280, out_channels=640, kernel_size=1, stride=1)
 
        # layer19 到 layer15  上采样，尺度大一倍
        self.upsample_960_960 = Upsample(19)
        self.conv_960_480 = nn.Conv2d(in_channels=960, out_channels=480, kernel_size=1, stride=1)
     
        # layer15 到 layer15 尺度不变
        #self.conv_96_96 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1,  stride=1)
        

        # 平滑层
        # self.smooth = DWConv2d(512, 512, kernel_size=3, padding=1, stride=1)
        # self.smooth1 = DWConv2d(1280, 1280, kernel_size=3, padding=1, stride=1)
        # self.smooth2 = DWConv2d(96, 96, kernel_size=3, padding=1, stride=1)

        self.smooth = CBL(576, 576, 1)
        self.smooth1 = CBL(640, 512, 1)
        self.smooth2 = CBL(960, 672, 1)
        # 通道数BN层的参数是输出通道数out_channels
        self.bn = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(640)
        self.bn3 = nn.BatchNorm2d(960)

        # CBAM模块【6个特征层：512 512 512 256 256 256 】
        # if USE_CBAM:
        #     self.CBAM1 = Bottleneck(512)
        #     self.CBAM2 = Bottleneck(1280)
        #     self.CBAM3 = Bottleneck(512)
        #     self.CBAM4 = Bottleneck(256)
        #     self.CBAM5 = Bottleneck(256)
        #     self.CBAM6 = Bottleneck(128)

        if USE_SE:
            self.SE1 = SEModule(576)
            self.SE2 = SEModule(672)
            self.SE3 = SEModule(512)
            self.SE4 = SEModule(256)
            self.SE5 = SEModule(256)
            self.SE6 = SEModule(128) 

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        
        # sources = list()
        # apply vgg up to conv4_3 relu
        x = self.mobilenet.conv1(x)
        x = self.mobilenet.bn1(x)
        x = self.mobilenet.activation(x)

        for i in self.mobilenet.bottlenecks[:5]:
            x = i(x)

        #s = self.L2Norm(x)
        #print(x.shape)
        sources.append(x)

        # apply vgg up to fc7

        for i in self.mobilenet.bottlenecks[5:]:
            x = i(x)
        x = self.mobilenet.conv_last(x)
        x = self.mobilenet.bn_last(x)
        x = self.mobilenet.activation(x)

        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            #print(x.size())
            #print(v(x).size())
            # x = F.relu(v(x), inplace=True)
            x = v(x)
            if k % 2 == 1:
                sources.append(x)
        # for x in sources:
        #     print(x.shape)

        sources_final = list()
        
        conv19_2_fp = self.SB_256(sources[3])
   
        # conv19_1层融合结果  self.bn1(self.conv1(x)) 在通道维度上融合
        conv19_1_fp1 = torch.cat((self.conv_256_128(self.bn(self.upsample_256_256(conv19_2_fp))), sources[2])
                               ,1)
        conv19_1_fp2 = self.SB_640(conv19_1_fp1)
      
        conv19_1_fp = self.smooth1(conv19_1_fp2)
  
        # conv19层融合结果
        conv19_fp1 =  torch.cat((self.conv_640_320(self.bn2(self.upsample_640_640(conv19_1_fp2))), self.conv_1280_640(sources[1]))
                               ,1)
        conv19_fp2 = self.SB_960(conv19_fp1)
        conv19_fp = self.smooth2(conv19_fp1)
   
        # conv15层融合结果
        conv15_fp = torch.cat(((self.conv_960_480(self.bn3(self.upsample_960_960(conv19_fp2)))), sources[0])
                               ,1)
        conv15_fp = self.smooth(conv15_fp)
        # if USE_CBAM:
        #     # print("use cbam")
        #     sources_final.append(self.CBAM1(conv15))
        #     sources_final.append(self.CBAM2(layer19))
        #     sources_final.append(self.CBAM3(conv19_1))
        #     sources_final.append(self.CBAM4(sources[3]))
        #     sources_final.append(self.CBAM5(sources[4]))
        #     sources_final.append(self.CBAM6(sources[5]))
        if USE_SE:
            sources_final.append(self.SE1(conv15_fp))
            sources_final.append(self.SE2(conv19_fp))
            sources_final.append(self.SE3(conv19_1_fp))
            sources_final.append(self.SE4(conv19_2_fp))
            sources_final.append(self.SE5(sources[4]))
            sources_final.append(self.SE6(sources[5]))

        else:
            #print("no cbam")
            sources_final.append(conv15_fp)
            sources_final.append(conv19_fp)
            sources_final.append(conv19_1_fp)
            sources_final.append(conv19_2_fp)
            sources_final.append(sources[4])
            sources_final.append(sources[5])
        # for x in sources_final:
        #     print(x.shape)
        # apply multibox head to source layers     
        for (x, l, c) in zip(sources_final, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        if self.phase == "test":
            # print(loc.shape, conf.shape)
            # print(self.priors.size())
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),                  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
            #print(output)
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location='cuda:0'))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def conv_dw(inp, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
    )

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv1_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def SB(n):
    return nn.Sequential(
        nn.Conv2d(in_channels=n, out_channels=n, kernel_size=1, stride=1),
        nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=3, padding=1, stride=1),
        nn.Conv2d(in_channels=2*n, out_channels=n, kernel_size=1, stride=1),
        nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=3, padding=1, stride=1),
        nn.Conv2d(in_channels=2*n, out_channels=n, kernel_size=1, stride=1),
        )

def CBL(inp, oup, stride):
    return nn.Sequential(
    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
    nn.BatchNorm2d(oup),
    nn.LeakyReLU(inplace=True))

#torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
def add_extras(i):
    # Extra layers added to VGG for feature scaling
    layers = []

    #conv14
    layers += [conv1_bn(i,256,1)]
    layers += [conv_bn(256,512,2)]

    #conv15
    layers += [conv1_bn(512,128,1)]
    layers += [conv_bn(128,256,2)]

    #con16
    layers += [conv1_bn(256,128,1)]
    layers += [conv_bn(128,256,2)]

    #conv17
    layers += [conv1_bn(256,64,1)]
    layers += [conv_bn(64,128,2)]

    return layers


def multibox(mobilenet, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []

    ### may be have bug ###
    #mobilenetv2_source = [5, -1]
    extras_source = [1,3,5,7]

    loc_layers += [nn.Conv2d(576, 4 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(576, 4 * num_classes, kernel_size=1)]

    loc_layers += [nn.Conv2d(672, 6 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(672, 6 * num_classes, kernel_size=1)]

    # for k, v in enumerate(extra_layers[1::2], 2):
    for k, v in enumerate(extras_source):
        k += 2
        loc_layers += [nn.Conv2d(extra_layers[v][0].out_channels,
                                 cfg[k] * 4, kernel_size=1)]
        conf_layers += [nn.Conv2d(extra_layers[v][0].out_channels,
                                  cfg[k] * num_classes, kernel_size=1)]
    return mobilenet, extra_layers, (loc_layers, conf_layers)

extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256],
    '512': [],
}

mbox = {
    '300':[4, 6, 6, 6, 6, 6],
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):

    # add, no use
    size = 300

    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    # base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
    #                                  add_extras(extras[str(size)], 1024),
    #                                  mbox[str(size)], num_classes)

    base_, extras_, head_ = multibox(mobilenetv2.MobileNet2(scale=1.0), add_extras(1280),mbox[str(size)], num_classes)

    return SSD(phase, size, base_, extras_, head_, num_classes)


if __name__ =="__main__":
    torch.backends.cudnn.enabled = False
    ssd = build_ssd("train")
    x = torch.zeros((32, 96, 19, 19))
    x = ssd.loc[0](x)
    print(x.size())
    x = torch.zeros((32, 1280, 10, 10))
    x = ssd.loc[1](x)
    print(x.size())
    x = torch.zeros((32, 512, 5, 5))
    x = ssd.loc[2](x)
    print(x.size())
    x = torch.zeros((32, 256, 3, 3))
    x = ssd.loc[3](x)
    print(x.size())
    x = torch.zeros((32, 256, 2, 2))
    x = ssd.loc[4](x)
    print(x.size())
    x = torch.zeros((32, 128, 1, 1))
    x = ssd.loc[5](x)
    print(x.size())


    # x = torch.zeros((32, 1280, 10, 10))
    #
    # for i in ssd.extras:
    #     x = i(x)
    #     print(x.size())

    # x = ssd.mobilenet.conv1(x)
    # print(x.size())
    # x = ssd.mobilenet.bn1(x)
    # print(x.size())
    # x = ssd.mobilenet.relu(x)
    # print(x.size())
    # x = ssd.mobilenet.layer1(x)
    # print(x.size())
    # x = ssd.mobilenet.layer2(x)
    # print(x.size())
    # x = ssd.mobilenet.layer3(x)
    # print(x.size())
    # x = ssd.mobilenet.layer4(x)
    # print(x.size())
    # x = ssd.mobilenet.layer5(x)
    # print(x.size())
    # x = ssd.mobilenet.layer6(x)
    # print(x.size())
    # x = ssd.mobilenet.layer7(x)
    # print(x.size())
    # x = ssd.mobilenet.layer8(x)
    # print(x.size())
    # x = ssd.mobilenet.conv9(x)
    # print(x.size())