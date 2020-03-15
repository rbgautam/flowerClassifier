#!/usr/bin/env python3
""" train_args.py
Part 2 of the Udacity AIPND final project submission for Craig Johnston.
predict_args.py contains the command line argument definitions for predict.py
"""

import argparse


def get_args():
    """
    Get argument parser for train cli.

    Command line argument examples:
    - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    - Choose architecture: python train.py data_dir --arch "vgg13"
    - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    - Use GPU for training: python train.py data_dir --gpu

    For argparse examples see https://pymotw.com/3/argparse
    Returns an argparse parser.
    """

    parser = argparse.ArgumentParser(
        description="Image prediction.",
        usage="python ./predict.py /path/to/image.jpg checkpoint.pth --top_k 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('path_to_image',
                        help='Path to image file.',
                        action="store")

    parser.add_argument('checkpoint_file',
                        help='Path to checkpoint file.',
                        action="store")

   
    parser.add_argument('--top_k',
                        action="store",
                        default=5,
                        dest='top_k',
                        type=int,
                        help='Return top KK most likely classes.',
                        )

    
    parser.parse_args()
    return parser


def main():
    """
        Main Function
    """
    print(f'USAGE: python ./predict.py /path/to/image.jpg checkpoint.pth  --top_k 5')


if __name__ == '__main__':
    main()
"""
 main() is called if script is executed on it's own.
"""
