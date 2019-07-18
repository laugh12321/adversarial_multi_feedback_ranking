'''
 _                       _     
| |                     | |    
| |     __ _ _   _  __ _| |__  
| |    / _` | | | |/ _` | '_ \ 
| |___| (_| | |_| | (_| | | | |
\_____/\__,_|\__,_|\__, |_| |_|
                    __/ |      
                   |___/           


update: 2019-07-18
@Author: Laugh                                                                     
'''
import os
import sys
import math
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool
from multiprocessing import cpu_count

from time import time
from time import strftime
from time import localtime

from Dataset import Dataset
from cli import parse_args
from sampling import *

if __name__ == '__main__':

    time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())

    # initilize arguments and logging
    args = parse_args(sys.argv[1:])
    init_logging(args, time_stamp)

    # initialize dataset
    dataset = Dataset(args.path + args.dataset)

    args.adver = 0
