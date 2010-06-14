from __future__ import with_statement
from sys import argv

with open(str(argv[1])) as f1:
    with open(str(argv[2])) as f2:
        counter , cont = 0, True
        while cont:
            a, b = f1.readline().split(' '), f2.readline().split(' ')
            for i in range(len(a)):
                if a[i] != b[i]:
                    print a, b,  counter
                    cont = False
                    break
            counter += 1
