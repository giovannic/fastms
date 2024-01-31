import argparse
from .sample import add_parser as add_sample_parser, run as run_sample
from .train import add_parser as add_train_parser, run as run_train
from .infer import add_parser as add_infer_parser, run as run_infer
from .validate import add_parser as add_validate_parser, run as run_validate
import logging
import os

parser = argparse.ArgumentParser(
    prog='fastms',
    description='Run surrogate modelling tasks for IC malaria models'
)

subparsers = parser.add_subparsers(dest='commands')
add_sample_parser(subparsers)
add_train_parser(subparsers)
add_infer_parser(subparsers)
add_validate_parser(subparsers)
args = parser.parse_args()

def run():
    logging.basicConfig(level=os.environ.get("FASTMS_LOG", "WARNING"))
    if args.commands == 'sample':
        run_sample(args)
    elif args.commands == 'train':
        run_train(args)
    elif args.commands == 'infer':
        run_infer(args)
    elif args.commands == 'validate':
        run_validate(args)
    else:
        raise NotImplementedError('Command not implemented yet')
