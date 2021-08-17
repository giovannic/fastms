import logging

FORMAT = '%(levelname)s: %(asctime)-15s %(message)s'

def setup_log(level):
    numeric_level = getattr(logging, level, None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % level)
    logging.basicConfig(level=numeric_level, format=FORMAT)
