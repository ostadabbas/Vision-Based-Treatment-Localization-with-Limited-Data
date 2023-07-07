from torch import nn

def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):

    '''
    Convolutional layer

    PARAMETERS:
        in_channels: Number of channels in the input tensor (int)
        out_channels: Number of channels in the output tensor (int)
        kernel_size: Kernel size (int)
        padding: Padding (int)
        bn: Whether to use batch normalization (bool)
        dilation: Dilation (int)
        stride: Stride (int)
        relu: Whether to use ReLU (bool)
        bias: Whether to use bias (bool)

    OUTPUT:
        nn.Sequential(): Convolutional layer
    '''
    
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):

    '''
    Depthwise convolutional layer

    PARAMETERS:
        in_channels: Number of channels in the input tensor (int)
        out_channels: Number of channels in the output tensor (int)
        kernel_size: Kernel size (int)
        padding: Padding (int)
        stride: Stride (int)
        dilation: Dilation (int)
        
    OUTPUT:
        nn.Sequential(): Depthwise convolutional layer
    '''

    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):

    '''
    Depthwise convolutional layer without batch normalization

    PARAMETERS:
        in_channels: Number of channels in the input tensor (int)
        out_channels: Number of channels in the output tensor (int)
        kernel_size: Kernel size (int)
        padding: Padding (int)
        stride: Stride (int)
        dilation: Dilation (int)
        
    OUTPUT:
        nn.Sequential(): Depthwise convolutional layer without batch normalization
    '''

    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.ELU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )
