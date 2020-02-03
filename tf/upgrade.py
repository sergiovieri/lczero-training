import argparse
import os
import yaml
import sys
import tensorflow as tf
from tfprocess import TFProcess


def find_weight(old_weights, weight):
    for w in old_weights:
        if weight.name == w.name:
            return w
    return None


def main(cmd):
    cfg_old = yaml.safe_load(cmd.cfg_old.read())
    cfg = yaml.safe_load(cmd.cfg.read())

    tfp_old = TFProcess(cfg_old)
    tfp_old.init_net_v2()
    tfp_old.restore_v2()
    old_weights = tfp_old.model.weights
    del tfp_old
    tf.keras.backend.clear_session()

    tfp = TFProcess(cfg)
    tfp.init_net_v2()
    for weight in tfp.model.weights:
        old_weight = find_weight(old_weights, weight)
        if old_weight is not None:
            weight.assign(old_weight)
        else:
            print('Weight', weight.name, 'not found')
    tfp.manager.save(checkpoint_number=0)
    import code
    code.interact(local=locals())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Convert current checkpoint to new training script or incompatible training parameters.')
    argparser.add_argument('--cfg_old', type=argparse.FileType('r'),
            help='yaml configuration with training parameters')
    argparser.add_argument('--cfg', type=argparse.FileType('r'),
            help='yaml configuration with training parameters')

    main(argparser.parse_args())
