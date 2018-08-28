import sys
import numpy
import argparse
import csv

TIMESTAMP_OFFSET = 5

def convert_csv(src_path, dst_path):

    csv_file = open(src_path, "r")
    reader = csv.reader(csv_file)

    w_csv_file = open(dst_path, "w")
    writer = csv.writer(w_csv_file)


    # result = ""

    for item in reader:

        if reader.line_num == 1:
            writer.writerow(item)
            continue
        # result += item[0] + " " + item[1] + "\n"

        # original: timestamp qx qy qz qwtx ty tz
        # tum : timestamp tx ty tz qx qy qz qw


        timestamp = str(int(item[0][TIMESTAMP_OFFSET:]))
        item[0] = timestamp

        # print(item)

        writer.writerow(item)

        # result += timestamp + " " + item[1] + " " + item[2] + " " + item[3] + " " + item[5] + " " + item[6] + " " + item[7] + " " + item[4] + "\n"

    csv_file.close()
    w_csv_file.close()
    # print(result)

    # tum_file = open(tum_list, "w")
    # tum_file.write(result)
    # tum_file.close()

if __name__=="__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script tries to convert the time stamp to our demo format. 
    ''')
    parser.add_argument('first_file', help='data.csv in state_groundtruth_estimate0/')
    parser.add_argument('second_file', help='file where you want to save the converted csv')

    args = parser.parse_args()

    first_file = args.first_file
    second_file = args.second_file

    convert_csv(first_file, second_file)
