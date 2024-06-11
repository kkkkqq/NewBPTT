
def get_model(modelname:str, num_classes:int=10, channel:int=3, image_size:tuple=(32,32), **kwargs):
    from models.convnet import ConvNet, convnet3
    from models.vaes.cvae import ConditionalVAE
    if modelname.lower()=='convnet3':
        return convnet3(channel=channel, num_classes=num_classes, image_size=image_size)
    elif modelname.lower()=='cvae':
        return ConditionalVAE(in_channels=channel,
                              num_classes=num_classes,
                              latent_dim=512,
                              img_size=image_size[0])
    else:
        raise NotImplementedError