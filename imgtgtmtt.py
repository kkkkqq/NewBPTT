import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from exps.imgtgtmtt_exp import ImgTgtMTTExp
import yaml
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generic runner for Training Synset')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/imgtgtmtt.yaml')
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    exp = ImgTgtMTTExp(config)
    exp.run_exp()
