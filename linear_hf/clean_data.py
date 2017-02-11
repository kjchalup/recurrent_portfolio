import os
import glob
import random
import joblib
import numpy as np
from datetime import datetime, timedelta
import fileinput

all_txt = glob.glob('tickerData/*.txt')
questionable_files = []
for fname in all_txt:
    
    filedata = None
    with open(fname, 'r') as file :
        filedata = file.read()
    f = open(fname, 'r').readlines()      
     
    if len([s for s in f if 'NaN,NaN,NaN,99' in s])>0: 
        # Replace the target string
        filedata = filedata.replace('NaN,NaN,NaN,99', 'NaN,NaN,NaN,NaN')
        with open(fname, 'w') as file:
            file.write(filedata)

        import pdb;pdb.set_trace()
    
   # Write the file out again
    
   
