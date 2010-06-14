from sys import argv

def convert_args():
    args = {}
    for arg in argv:
        if arg.find('=') > -1:
            index = arg.find('=')
            args[arg[:index]] = arg[index + 1:]
        else:
            args[arg] = True
    return args

args = convert_args()

f = open(args['file'], 'r')
data, temp = [], f.read().split('\n')
tresh = float(args['tresh'])
for i in range(len(temp[:-1])):
    item = temp[i]
    s = item.split(',')
    if float(s[2]) > tresh:
       print s
       exit()
