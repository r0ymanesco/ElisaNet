import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train-batch-size', required=True, type=int,
                    help='train batch size')
parser.add_argument('--eval-batch-size', required=True, type=int,
                    help='eval batch size')
parser.add_argument('--gpu', required=True, type=int,
                    help='GPU index to run on')
parser.add_argument('--lr', required=True, type=float,
                    help='learning rate')
parser.add_argument('--model-dir', type=str, default='model',
                    help='Path to model folder.')
parser.add_argument('--c_feat', required=True, type=int,
                    help='number of feature channels')
parser.add_argument('--epochs', required=True, type=int,
                    help='number of epochs to run')
parser.add_argument('--resume_epoch', required=True, type=int,
                    help='epoch to resume training at')
parser.add_argument('--lr_resume', required=False, type=float,
                    help='learning rate to resume to')
parser.add_argument('--patience', required=True, type=int,
                    help='early stopping patience')
parser.add_argument('--loss-delta', required=True, type=float,
                    help='early stopping loss delta (can be max or min)')
parser.add_argument('--es-mode', required=True, type=str,
                    help='early stopping mode (can be max or min)')
parser.add_argument('--eval', default=False, type=bool,
                    help='whether to go into eval mode directly')

parser.parse_args()
