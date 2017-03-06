""" Clean data of 'NaN,NaN,NaN,99' for open, high, low, close data.

    Replaces strings of 'NaN,NaN,NaN,99' to 'NaN,NaN,NaN,NaN'. The "99 pattern"
    happens in CRSP data when a stock is delisted.
"""
import os
import glob
DATA_DIR = 'tickerData'

if __name__ == "__main__":
    ALL_TXT = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    QUESTIONABLE_FILES = []
    for fname in ALL_TXT:
        filedata = None
        with open(fname, 'r') as f:
            filedata = f.read()
        filedata = filedata.replace('NaN,NaN,NaN,99', 'NaN,NaN,NaN,NaN')
        with open(fname, 'w') as f:
            f.write(filedata)
