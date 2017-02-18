import glob

""" Clean data of 'NaN,NaN,NaN,99' for open, high, low, close data.

    Replaces strings of 'NaN,NaN,NaN,99' to 'NaN,NaN,NaN,NaN'.
"""

all_txt = glob.glob('tickerData/*.txt')
questionable_files = []
for fname in all_txt:

    filedata = None
    with open(fname, 'r') as file:
        filedata = file.read()
    f = open(fname, 'r').readlines()

    if len([s for s in f if 'NaN,NaN,NaN,99' in s])>0:
        # Replace the target string
        filedata = filedata.replace('NaN,NaN,NaN,99', 'NaN,NaN,NaN,NaN')
        with open(fname, 'w') as file:
            file.write(filedata)
