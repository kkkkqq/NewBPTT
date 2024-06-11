def get_dataset(dataset_name:str, data_path:str, *args, **kwargs):
    from dataset.cifar10 import CIFAR10
    if dataset_name.lower() == 'cifar10':
        return CIFAR10(data_path=data_path, *args, **kwargs)
    else:
        raise NotImplementedError