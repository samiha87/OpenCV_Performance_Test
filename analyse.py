import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Select input")
ap.add_argument("-t", "--trheshold", required=True, help="Select ratio_treshold")

args = vars(ap.parse_args())
# Read file, file path from args
f = open (args["input"], "r")
comment_lines = 0
code_lines = 0
filter_list = []

for x in f:
    if "SUM:" in x:
        # split strings
        sumstring = x.split(' ')
        for z in sumstring:
            # Filter empty lines
            if z != '':
                filter_list.append(z)
comment_lines = float(filter_list[3])
code_lines = float(filter_list[4])
# Calculate comment code ratio
ratio = comment_lines / code_lines
# Check if calculated ratio is higher than set treshold
if ratio >= float(args["trheshold"]):
    print("Ok")
else:
    print("Fail")

sys.exit(0)
