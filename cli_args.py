

import argparse

"""
Supported architectures.
"""
supported_arch = [
    'vgg11',
    'vgg13',
    'vgg16'
    
]


def get_args():
    """
    Get arguments from commandline parser for train module

    Command line argument examples:
    USAGE:
    python train.py data_dir --save_dir save_directory --arch "vgg13" --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu
   
    For argparse examples see https://pymotw.com/3/argparse
    Returns an argparse parser.
    """

    parser = argparse.ArgumentParser(
        description="Train and save an image classifier model.",
        usage="python ./train.py ./flowers/train --gpu --learning_rate 0.001 --hidden_units 3136 --epochs 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('data_directory', action="store")

    parser.add_argument('--save_dir',
                        action="store",
                        default=".",
                        dest='save_dir',
                        type=str,
                        help='Directory to save training model file',
                        )

    parser.add_argument('--save_name',
                        action="store",
                        default="checkpoint",
                        dest='save_name',
                        type=str,
                        help='Model filename.',
                        )

    parser.add_argument('--categories_json',
                        action="store",
                        default="cat_to_name.json",
                        dest='categories_json',
                        type=str,
                        help='Path to JSON file containing the labels.',
                        )

    parser.add_argument('--arch',
                        action="store",
                        default="vgg16",
                        dest='arch',
                        type=str,
                        help='Supported architectures: ' + ", ".join(supported_arch),
                        )

    parser.add_argument('--gpu',
                        action="store_true",
                        dest="use_gpu",
                        default=False,
                        help='Use GPU')

    hp = parser.add_argument_group('hyperparameters')

    hp.add_argument('--learning_rate',
                    action="store",
                    default=0.001,
                    type=float,
                    help='Learning rate')

    hp.add_argument('--hidden_units', '-hu',
                    action="store",
                    dest="hidden_units",
                    default=[3136, 784],
                    type=int,
                    nargs='+',
                    help='Hidden layer units')

    hp.add_argument('--epochs',
                    action="store",
                    dest="epochs",
                    default=1,
                    type=int,
                    help='Epochs')

    parser.parse_args()
    return parser


