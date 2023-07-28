def add_parser(subparsers):
    """add_parser. Adds the inference parser to the main ArgumentParser
    :param subparsers: the subparsers to modify
    """
    sample_parser = subparsers.add_parser(
        'infer',
        help='Infers model parameters'
    )
    sample_parser.add_argument(
        'target',
        choices=['EIR', 'intrinsic'],
        help='Parameters to infer'
    )
    sample_parser.add_argument(
        'model',
        choices=['eq', 'det', 'ibm'],
        help='Model to use for inference'
    )
    sample_parser.add_argument(
        'observations',
        type=str,
        help='Path to observational data'
    )
    sample_parser.add_argument(
        'output',
        type=str,
        help='Path to save the posterior estimates in'
    )
    sample_parser.add_argument(
        '--surrogate',
        type=str,
        help='Path to a surrogate model to use'
    )
