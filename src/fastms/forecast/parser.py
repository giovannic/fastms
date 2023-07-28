def add_parser(subparsers):
    """add_parser. Adds the inference parser to the main ArgumentParser
    :param subparsers: the subparsers to modify
    """
    sample_parser = subparsers.add_parser(
        'forecast',
        help='Evaluate the forecasts made using the surrogate'
    )
    sample_parser.add_argument(
        'surrogate',
        type=str,
        help='Path to the surrogate to use for forecasting'
    )
    sample_parser.add_argument(
        'observations',
        type=str,
        help='Path to observations to use to assess forecast accuracy'
    )
