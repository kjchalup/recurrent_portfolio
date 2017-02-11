import os
import glob
import random
import joblib
import numpy as np
from datetime import datetime, timedelta


def test_clean_files():

    all_txt = glob.glob('tickerData/*.txt')

    questionable_files = []
    
    for fname in all_txt:
        f = open(fname, 'r').readlines()
        lines=[s for s in f if 'NaN,NaN,NaN,99' in s]
        if len(lines)>0:
            questionable_files.append(fname)
    
    print("This many dirty files: "+str(len(questionable_files)))
    assert questionable_files is None, "Dirty files found! Run linear_hf/clean_data.py"
    

