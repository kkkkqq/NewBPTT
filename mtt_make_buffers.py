import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import yaml
from mtt_related.make_buffers import make_buffers


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generic runner for Training Synset')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='mtt_related/buffer_config.yaml')
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    make_buffers(**config)