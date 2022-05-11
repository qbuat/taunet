import argparse



common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument('--debug', default=False, action='store_true')
common_parser.add_argument('--nfiles', default=-1, type=int)
prongs = common_parser.add_mutually_exclusive_group()
prongs.add_argument('--1prong', default=False, action='store_true')
prongs.add_argument('--3prongs', default=False,  action='store_true')


train_parser = argparse.ArgumentParser(parents=[common_parser])
train_parser.add_argument('--use-cache', action='store_true')

plot_parser = argparse.ArgumentParser(parents=[common_parser])
plot_parser.add_argument('--model', default='simple_dnn.h5')

