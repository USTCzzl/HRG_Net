def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    elif network_name == 'unet':
        from .unet import Unet
        return Unet
    elif network_name == 'resnet':
        from .resnet import GenerativeResnet
        return  GenerativeResnet
    elif network_name=="hrnet":
        from .HEHERnet_official import HRNet
        return HRNet
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
