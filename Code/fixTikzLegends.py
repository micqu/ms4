import sys
import fileinput, re
import ntpath

def modify_file(file_name, pattern, lambda_value_change):
    inputfile = open(file_name, 'r').readlines()
    write_file = open(file_name,'w')
    i = 0
    for line in inputfile:
        if pattern in line:
            write_file.write(lambda_value_change(line, i))
            i += 1
        else:
            write_file.write(line)
    write_file.close()

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

file_paths = sys.argv[1:]  # the first argument is the script itself

for p in file_paths:
    filename = path_leaf(p)
    print(filename)

    epochs = False
    if epochs:
        a = r"\addlegendentry{FA DP E=15}"
        b = [
            r"\addlegendentry{Central B=16}",
            r"\addlegendentry{FA E=5}",
            r"\addlegendentry{FA DP E=5}",
            r"\addlegendentry{FA E=15}",
            r"\addlegendentry{FA DP E=15}",
        ]
    else:
        a = r"\addlegendentry{FA DP B=64}"
        b = [
            r"\addlegendentry{Central B=16}",
            r"\addlegendentry{FA B=16}",
            r"\addlegendentry{FA DP B=16}",
            r"\addlegendentry{FA B=64}",
            r"\addlegendentry{FA DP B=64}",
        ]
    modify_file(p, a, lambda x,i: f"{b[i]}\n")
# input("Press Enter to continue...")