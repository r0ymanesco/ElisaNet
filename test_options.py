import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--images', type=str, required=True,
                    help='Path to test images. Separate multiple files using , without spaces.')
parser.add_argument('--gpu', required=False, type=int,
                    help='GPU index to run on if using GPU')

parser.parse_args()
