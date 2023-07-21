def add_parser(subparsers):
    """add_parser. Adds the sample parser to the main ArgumentParser
    :param subparsers: the subparsers to modify
    """
    sample_parser = subparsers.add_parser(
        'sample',
        help='Samples input parameters and model outputs for training'
    )
    sample_parser.add_argument(
        'model',
        choices=['eq', 'det', 'ibm'],
        help='Model to sample from'
    )
    sample_parser.add_argument(
        'intrinsic_strategy',
        choices=['prior', 'lhs'],
        help='Strategy for modelling intrinsic parameters'
    )
    sample_parser.add_argument(
        'output',
        type=str,
        help='Path for the samples to be saved in'
    )
    sample_parser.add_argument(
        '--sites',
        type=str,
        help='Path to site parameters. If not set, sites are sampled using the ' + 
            'LHS strategy'
    )
    sample_parser.add_argument(
        '--cores',
        type=int,
        help='Number of cores to use'
    )
