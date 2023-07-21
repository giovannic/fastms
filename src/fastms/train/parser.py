def add_parser(subparsers):
    """add_parser. Adds the training parser to the main ArgumentParser
    :param subparsers: the subparsers to modify
    """
    sample_parser = subparsers.add_parser(
        'train',
        help='Trains a surrogate model'
    )
    sample_parser.add_argument(
        'model',
        choices=['mlp', 'rnn', 'transformer'],
        help='Surrogate model to use'
    )
    sample_parser.add_argument(
        'samples',
        type=str,
        help='Path for the samples to use for training'
    )
    sample_parser.add_argument(
        'output',
        type=str,
        help='Path to save the model in'
    )
