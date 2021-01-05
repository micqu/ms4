import sys
import fileinput, re
import ntpath

def modify_file(file_name, pattern, lambda_value_change):
    inputfile = open(file_name, 'r').readlines()
    write_file = open(file_name,'w')
    for line in inputfile:
        if pattern in line:
            write_file.write(lambda_value_change(line))
        else:
            write_file.write(line)
    write_file.close()

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

file_paths = sys.argv[1:]  # the first argument is the script itself

ex_number = input("Type ex number: ")

for p in file_paths:
    a = r"% This file was created by tikzplotlib v0.9.6."
    b = r"\begin{axis}["
    c = "height=7cm,\nwidth=0.46\\textwidth,"
    filename = path_leaf(p)

    # print(a)
    # print(b)
    # print(c)
    print(filename)

    modify_file(p, a, lambda x: x + "\\tikzsetnextfilename{ex" + ex_number + "_" + filename[:-4] + "}\n")
    modify_file(p, b, lambda x: x + f"{c}\n")

# input("Press Enter to continue...")