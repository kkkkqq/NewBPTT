def make_buffers(dataset_args:dict,
                 model_args:dict, 
                 opt_args:dict, 
                 expert_module_args:dict,
                 buffer_folder_name:str,
                 repeat:int=100,
                 max_epoch:int=25, 
                 buffer_interval:int=3):
    from os import makedirs
    from os.path import join, exists
    from models.utils import get_model
    from utils import get_optimizer
    from modules.utils import get_module
    from dataset.utils import get_dataset
    from torch import save
    from torch.utils.data import DataLoader
    dataset = get_dataset(**dataset_args)
    real_loader = DataLoader(dataset.dst_train, 1024, True, pin_memory=True, num_workers=4)
    module = get_module(**expert_module_args)
    bufferfolder = buffer_folder_name
    if not exists(bufferfolder):
        makedirs(bufferfolder)
    for expert_idx in range(repeat):
        model = get_model(**model_args)
        model.to('cuda')
        opt = get_optimizer(model.parameters(), **opt_args)
        for it in range(max_epoch):
            if it%buffer_interval==0:
                model_stdt = model.state_dict()#params stored on cuda
                opt_stdt = opt.state_dict()#states stored on cuda
                save_dct = {'model': model_stdt, 'optimizer': opt_stdt}
                buffername = ''.join(['exp', str(expert_idx), '_', 'epoch',str(it),'.pt'])
                save(save_dct, join(bufferfolder, buffername))
                print('saved', join(bufferfolder, buffername))
            module.epoch(model,
                        opt,
                        real_loader,
                        False,
                        True)
                


        
    
    
