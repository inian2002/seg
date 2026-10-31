# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------
import logging
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

BatchNorm3d       = nn.BatchNorm3d
BatchNorm3d_class = nn.BatchNorm3d
BN_MOMENTUM       = 0.1
logger            = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 8

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = BatchNorm3d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1      = conv3x3(inplanes, planes, stride)
        self.bn1        = norm_layer(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = conv3x3(planes, planes)
        self.bn2        = norm_layer(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class MCF_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MCF_Block, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.relu         = nn.ReLU()
        self.conv         = nn.Conv3d(in_channels, in_channels, 1)
        self.skip_conv    = nn.Conv3d(in_channels, out_channels, 1)
        self.compre_conv  = nn.Conv3d(in_channels, out_channels, 1)
        self.bn           = nn.BatchNorm3d(in_channels)
        self.norm_conv    = nn.Conv3d(in_channels, in_channels, 3, padding=1)
        self.branch_conv1 = nn.Conv3d(in_channels, in_channels // 2,
                                      3, padding=2, dilation=2)
        self.branch_conv2 = nn.Conv3d(in_channels // 2, in_channels, 1)

    def forward(self, input):
        x     = self.relu(self.bn(self.conv(input)))
        x     = self.norm_conv(x)
        w     = torch.sigmoid(self.branch_conv2(self.branch_conv1(x)))
        x     = self.compre_conv(x * w)
        out   = x + self.skip_conv(input)
        return out


class HighResolutionModule3D(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule3D, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks,
                             num_inchannels, num_channels)
        self.num_inchannels    = num_inchannels
        self.fuse_method       = fuse_method
        self.num_branches      = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches          = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu        = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        for name, lst in [('NUM_BLOCKS', num_blocks),
                          ('NUM_CHANNELS', num_channels),
                          ('NUM_INCHANNELS', num_inchannels)]:
            if num_branches != len(lst):
                msg = f'NUM_BRANCHES({num_branches}) <> {name}({len(lst)})'
                logger.error(msg); raise ValueError(msg)

    def _make_one_branch(self, branch_index, block, num_blocks,
                         num_channels, stride=1):
        downsample = None
        if (stride != 1 or
                self.num_inchannels[branch_index] !=
                num_channels[branch_index] * block.expansion):
            downsample = nn.Sequential(
                nn.Conv3d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm3d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM))
        layers = [block(self.num_inchannels[branch_index],
                        num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([
            self._make_one_branch(i, block, num_blocks, num_channels)
            for i in range(num_branches)
        ])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches   = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers    = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv3d(num_inchannels[j], num_inchannels[i],
                                  1, 1, 0, bias=False),
                        BatchNorm3d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv3d(num_inchannels[j], num_inchannels[i],
                                          3, 2, 1, bias=False),
                                BatchNorm3d(num_inchannels[i],
                                            momentum=BN_MOMENTUM)))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv3d(num_inchannels[j], num_inchannels[j],
                                          3, 2, 1, bias=False),
                                BatchNorm3d(num_inchannels[j],
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    w = x[i].shape[-1]; h = x[i].shape[-2]; d = x[i].shape[-3]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[d, h, w], mode='trilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class SaltHRNet3D(nn.Module):
    """3D high-resolution network for salt/fault segmentation.

    The constructor and forward contract are shared by the training and
    inference entry points in this project.
    """
    def __init__(self, base=8, enable_reconstruction=True):
        super(SaltHRNet3D, self).__init__()
        self.base    = base
        self.enable_reconstruction = enable_reconstruction
        nc           = base
        self.conv1   = nn.Sequential(
            nn.Conv3d(1, nc, kernel_size=3, stride=2, padding=1),
            BatchNorm3d(nc, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.conv1_0 = BasicBlock(nc, nc, 1)
        self.conv1_1 = nn.Conv3d(nc, nc, kernel_size=3, stride=2,
                                 padding=1, bias=False)
        self.bn1_1   = BatchNorm3d(nc, momentum=BN_MOMENTUM)
        self.relu    = nn.ReLU(inplace=True)

        self.layer1          = self._make_layer(Bottleneck, nc, nc, 1)
        stage1_out           = Bottleneck.expansion * nc
        ch2                  = [base * i * BasicBlock.expansion
                                 for i in range(1, 3)]
        self.transition1     = self._make_transition_layer([stage1_out], ch2)
        self.stage2, pre2    = self._make_stage(ch2, 2)
        ch3                  = [base * i * BasicBlock.expansion
                                 for i in range(1, 4)]
        self.transition2     = self._make_transition_layer(pre2, ch3)
        self.stage3, _       = self._make_stage(ch3, 3)

        block = BasicBlock
        self.fuse1      = nn.Conv3d(base,     base, 1)
        self.fuse2      = nn.Conv3d(base * 2, base, 1)
        self.fuse3      = nn.Conv3d(base * 3, base, 1)
        self.fuse_layer1 = MCF_Block(base * 3, base)
        self.fuse_layer2 = MCF_Block(base * 2, base)
        self.layer2      = nn.Sequential(block(base, base, 1),
                                         block(base, base, 1))
        self.last_layer  = nn.Sequential(
            block(base, base, 1),
            nn.Conv3d(base, 1, kernel_size=1, stride=1, padding=0))
        if self.enable_reconstruction:
            self.recon_head = nn.Sequential(
                block(base, base, 1),
                nn.Conv3d(base, base, kernel_size=3, stride=1, padding=1),
                BatchNorm3d(base, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv3d(base, 1, kernel_size=1, stride=1, padding=0))
        else:
            self.recon_head = None

    def _make_transition_layer(self, pre, cur):
        layers = []
        for i in range(len(cur)):
            if i < len(pre):
                if cur[i] != pre[i]:
                    layers.append(nn.Sequential(
                        nn.Conv3d(pre[i], cur[i], 3, 1, 1, bias=False),
                        BatchNorm3d(cur[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    layers.append(None)
            else:
                convs = []
                for j in range(i + 1 - len(pre)):
                    ic = pre[-1]
                    oc = cur[i] if j == i - len(pre) else ic
                    convs.append(nn.Sequential(
                        nn.Conv3d(ic, oc, 3, 2, 1, bias=False),
                        BatchNorm3d(oc, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                layers.append(nn.Sequential(*convs))
        return nn.ModuleList(layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm3d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, num_inchannels, num_branches,
                    multi_scale_output=True):
        num_modules  = 2
        num_blocks   = [2] * num_branches
        num_channels = [self.base * i for i in range(1, num_branches + 1)]
        modules      = []
        for i in range(num_modules):
            mso = True if (multi_scale_output or i < num_modules - 1) else False
            modules.append(HighResolutionModule3D(
                num_branches, BasicBlock, num_blocks,
                num_inchannels, num_channels, 'SUM', mso))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def _forward_features(self, x):
        x  = self.conv1(x)
        x  = self.conv1_0(x);  x0 = x
        x  = self.relu(self.bn1_1(self.conv1_1(x)))
        x  = self.layer1(x)

        xl = []
        for i in range(2):
            xl.append(self.transition1[i](x) if self.transition1[i] else x)
        yl = self.stage2(xl)

        xl = []
        for i in range(3):
            if self.transition2[i]:
                xl.append(self.transition2[i](yl[i] if i < 2 else yl[-1]))
            else:
                xl.append(yl[i])
        x = self.stage3(xl)

        x1 = self.fuse1(x[0])
        b, c, w, h, d = x1.shape
        x2 = F.interpolate(self.fuse2(x[1]), size=(w, h, d),
                           mode='trilinear', align_corners=True)
        x3 = F.interpolate(self.fuse3(x[2]), size=(w, h, d),
                           mode='trilinear', align_corners=True)
        x  = self.fuse_layer1(torch.cat([x1, x2, x3], dim=1))
        x  = F.interpolate(x, scale_factor=(2, 2, 2),
                           mode='trilinear', align_corners=True)
        x  = self.fuse_layer2(torch.cat((x, x0), dim=1))
        x  = self.layer2(x)
        return F.interpolate(x, scale_factor=(2, 2, 2),
                             mode='trilinear', align_corners=True)

    def forward(self, x, return_aux=False):
        feat = self._forward_features(x)
        seg_logits = self.last_layer(feat)
        seg = torch.sigmoid(seg_logits)

        if not return_aux:
            return seg

        recon = self.recon_head(feat) if self.recon_head is not None else None
        return {
            'seg': seg,
            'recon': recon,
        }
__all__ = [
    'SaltHRNet3D',
]
