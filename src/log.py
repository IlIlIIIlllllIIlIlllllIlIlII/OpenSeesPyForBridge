
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
shandler = logging.StreamHandler()
format = logging.Formatter('%(name)s %(asctime)s {%(levelname)s}:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
shandler.setFormatter(format)
logger.addHandler(shandler)