import argparse
from .sample.parser import add_parser as add_sample_parser
from .train.parser import add_parser as add_train_parser
from .infer.parser import add_parser as add_infer_parser
from .forecast.parser import add_parser as add_forecast_parser

parser = argparse.ArgumentParser(
    prog='fastms',
    description='Run surrogate modelling tasks for IC malaria models'
)
subparsers = parser.add_subparsers(dest='commands')
add_sample_parser(subparsers)
add_train_parser(subparsers)
add_infer_parser(subparsers)
add_forecast_parser(subparsers)
args = parser.parse_args()
