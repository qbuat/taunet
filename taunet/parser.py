"""Command line arguments for fitting and plotting"""

import argparse
from tokenize import Double

common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument('--debug', default=False, action='store_true', help='Use only a portion of data from dataset')
common_parser.add_argument('--nfiles', default=-1, type=int, help='Number of files to be used from complete dataset')
common_parser.add_argument('--no-norm-target', default=False, action='store_true', help='Do not normalize target (default: False)')
common_parser.add_argument('--no-normalize', default=False, action='store_true', help='Do not normalize any variables (default: False')
common_parser.add_argument('--use-cache', default=False, action='store_true', help='Use previously cache data instead of importing dataset')
common_parser.add_argument('--add-to-cache', default=False, action='store_true', help='Add pre-processed data to cache')
common_parser.add_argument('--newTarget', default=False, action='store_true', help='Regress on truth/calo instead of truth/combined')
prongs = common_parser.add_mutually_exclusive_group()
prongs.add_argument('--oneProng', default=False, action='store_true')
prongs.add_argument('--threeProngs', default=False,  action='store_true')

# parse commands for training (fit.py)
train_parser = argparse.ArgumentParser(parents=[common_parser])
train_parser.add_argument('--rate', default=1e-5, type=float)
train_parser.add_argument('--batch-size', default=64, type=int)
train_parser.add_argument('--big-2gauss', default=False, action='store_true')
train_parser.add_argument('--small-2gauss', default=False, action='store_true')
train_parser.add_argument('--small-1gauss', default=False, action='store_true')
train_parser.add_argument('--small-2gauss-noreg', default=False, action='store_true')
train_parser.add_argument('--gauss3', default=False, action='store_true')

# parse commands for plotting (plot.py)
plot_parser = argparse.ArgumentParser(parents=[common_parser])
plot_parser.add_argument('--model', default='gauss2_simple_mdn_noreg.h5')
plot_parser.add_argument('--copy-to-cernbox', default=False, action='store_true')
plot_parser.add_argument('--path', default='')
plot_parser.add_argument('--get-above-below', default=False, action='store_true', help='Get events above and below |std/mean| = 1')
plot_parser.add_argument('--get-GMM-components', default=False, action='store_true', help='Get gaussian mixture model components')
