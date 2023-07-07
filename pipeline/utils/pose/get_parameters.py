from torch import nn

def get_parameters(model, predicate):

    '''
    Get parameters of a model

    PARAMETERS:
        model: Model (nn.Module)
        predicate: Predicate (function)
        
    OUTPUT:
        parameters: Parameters (generator)
    '''

    for module in model.modules():
        for param_name, param in module.named_parameters():
            if predicate(module, param_name):
                yield param


def get_parameters_conv(model, name):

    '''
    Get parameters of a convolutional layer

    PARAMETERS:
        model: Model (nn.Module)
        name: Name of the parameter (str)

        
    OUTPUT:
        parameters: Parameters (generator)
    '''

    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2d) and m.groups == 1 and p == name)


def get_parameters_conv_depthwise(model, name):

    '''
    Get parameters of a depthwise convolutional layer

    PARAMETERS:
        model: Model (nn.Module)
        name: Name of the parameter (str)
        
    OUTPUT:
        parameters: Parameters (generator)
    '''

    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2d)
                                              and m.groups == m.in_channels
                                              and m.in_channels == m.out_channels
                                              and p == name)


def get_parameters_bn(model, name):

    '''
    Get parameters of a batch normalization layer

    PARAMETERS:
        model: Model (nn.Module)
        name: Name of the parameter (str)
        
    OUTPUT:
        parameters: Parameters (generator)
    '''

    return get_parameters(model, lambda m, p: isinstance(m, nn.BatchNorm2d) and p == name)
