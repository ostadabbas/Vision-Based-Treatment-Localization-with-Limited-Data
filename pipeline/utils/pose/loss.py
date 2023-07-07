def l2_loss(input, target, mask, batch_size):

    '''
    L2 loss

    PARAMETERS:
        input: Input (tensor)
        target: Target (tensor)
        mask: Mask (tensor)
        batch_size: Batch size (int)
        
    OUTPUT:
        loss: Loss (tensor)
    '''

    loss = (input - target) * mask
    loss = (loss * loss) / 2 / batch_size

    return loss.sum()
