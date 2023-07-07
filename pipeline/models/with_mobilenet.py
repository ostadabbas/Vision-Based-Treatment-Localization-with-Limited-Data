import torch
from torch import nn

from utils.pose.conv import conv, conv_dw, conv_dw_no_bn

class Cpm(nn.Module):

    def __init__(self, in_channels, out_channels):
        
        '''
        CPM module

        PARAMETERS:
            in_channels: Number of channels in the input tensor (int)
            out_channels: Number of channels in the output tensor (int)

        OUTPUT:
            None
        '''
        
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):

        '''
        Forward pass of the CPM module

        PARAMETERS:
            x: Input tensor (torch.Tensor)

        OUTPUT:
            x: Output tensor (torch.Tensor)
        '''

        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):

    def __init__(self, num_channels, num_heatmaps, num_pafs):
        '''
        Initial stage of the network

        PARAMETERS:
            num_channels: Number of channels in the input tensor (int)
            num_heatmaps: Number of heatmaps to be predicted (int)
            num_pafs: Number of PAFs to be predicted (int)

        OUTPUT:
            None
        '''

        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):

        '''
        Forward pass of the initial stage

        PARAMETERS:
            x: Input tensor (torch.Tensor)

        OUTPUT:
            heatmaps: Heatmaps (torch.Tensor)
            pafs: PAFs (torch.Tensor)
        '''

        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]

class RefinementStageBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        
        '''
        Refinement stage block

        PARAMETERS:
            in_channels: Number of channels in the input tensor (int)
            out_channels: Number of channels in the output tensor (int)

        OUTPUT:
            None
        '''

        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):

        '''
        Forward pass of the refinement stage block

        PARAMETERS:
            x: Input tensor (torch.Tensor)

        OUTPUT:
            x: Output tensor (torch.Tensor)
        '''

        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):

    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):

        '''
        Refinement stage of the network

        PARAMETERS:
            in_channels: Number of channels in the input tensor (int)
            out_channels: Number of channels in the output tensor (int)
            num_heatmaps: Number of heatmaps to be predicted (int)
            num_pafs: Number of PAFs to be predicted (int)

        OUTPUT:
            None
        '''

        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):

        '''
        Forward pass of the refinement stage

        PARAMETERS:
            x: Input tensor (torch.Tensor)

        OUTPUT:
            heatmaps: Heatmaps (torch.Tensor)
            pafs: PAFs (torch.Tensor)
        '''
        
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class PoseEstimationWithMobileNet(nn.Module):

    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):

        '''
        Pose estimation network with MobileNet backbone

        PARAMETERS:
            num_refinement_stages: Number of refinement stages (int)
            num_channels: Number of channels in the input tensor (int)
            num_heatmaps: Number of heatmaps to be predicted (int)
            num_pafs: Number of PAFs to be predicted (int)

        OUTPUT:
            None
        '''

        super().__init__()
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        
        '''
        Forward pass of the network

        PARAMETERS:
            x: Input tensor (torch.Tensor)

        OUTPUT:
            stages_output: List of heatmaps and PAFs (list)
        '''

        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output
