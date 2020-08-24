import random
import logging
import datetime
import os


def setup_logger(name=""):
    my_logger = logging.getLogger("testLogger")
    my_logger.setLevel(logging.DEBUG)

    c_handler = logging.StreamHandler()

    if not os.path.isdir("log"):
        os.mkdir("log")

    time_string = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
    f_handler = logging.FileHandler(f'log/{name}_at_{time_string}_{random.randint(1,10000)}.log')

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    c_format = logging.Formatter('%(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    my_logger.addHandler(c_handler)
    my_logger.addHandler(f_handler)

    return my_logger
