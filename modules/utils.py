

def get_module(module_name:str, module_args:dict):
    from modules.basemodule import BaseModule
    from modules.clfmodule import ClassifierModule
    from modules.clflbrtgtmodule import ClassifierLbrTgtModule
    from modules.vaemodule import CVAEModule
    if module_name.lower() in ['clfmodule', 'classifiermodule', 'classifier', 'clf']:
        return ClassifierModule(**module_args)
    if module_name.lower() in ['clflbrtgt', 'clflbrtgtmodule', 'classifierLbrTgtmodule', 'classifierLbrTgt']:
        return ClassifierLbrTgtModule(**module_args)
    elif module_name.lower() in ['cvae', 'cvaemodule']:
        return CVAEModule(**module_args)
    else:
        raise NotImplementedError