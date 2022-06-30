import argparse
from tokenize import Double

common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument('--debug', default=False, action='store_true')
common_parser.add_argument('--nfiles', default=-1, type=int)
prongs = common_parser.add_mutually_exclusive_group()
prongs.add_argument('--1prong', default=False, action='store_true')
prongs.add_argument('--3prongs', default=False,  action='store_true')

# parse commands for training (fit.py)
train_parser = argparse.ArgumentParser(parents=[common_parser])
train_parser.add_argument('--use-cache', action='store_true')
train_parser.add_argument('--rate', default=1e-5, type=float)
train_parser.add_argument('--batch-size', default=64, type=int)

# parse commands for plotting (plot.py)
plot_parser = argparse.ArgumentParser(parents=[common_parser])
plot_parser.add_argument('--model', default='simple_dnn.h5')
plot_parser.add_argument('--copy-to-cernbox', default=False, action='store_true')
plot_parser.add_argument('--path', default='')
plot_parser.add_argument('--use-cache', default=False, action='store_true')
plot_parser.add_argument('--add-to-cache', default=False, action='store_true')
