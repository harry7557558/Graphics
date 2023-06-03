#!/usr/bin/env python3

import subprocess
import os
import sys

argv = sys.argv

if len(argv) > 1:
    cwd = os.path.abspath(argv[1])
else:
    cwd = os.path.abspath('./')
cwd = cwd.replace('\\', '/')
if not cwd.endswith('/'):
    cwd += '/'

# get a list of files
files = subprocess.run(
    ['git', 'ls-tree', '--full-tree', '--name-only', '-r', 'HEAD'],
    cwd=cwd, stdout=subprocess.PIPE
)
files = files.stdout.decode('utf-8').strip().split('\n')
bins = ['json', 'txt', 'svg', 'stl', 'ply', 'obj', 'step', 'ipynb']
#bins = []
files = [f for f in files if f[f.rfind('.')+1:] not in bins]

# get file sizes
for i in range(len(files)):
    filename = cwd + files[i]
    try:
    	if not os.path.isfile(filename):
    	    size = 0
    	else:
            #size = os.stat(files[i]).st_size
            size = len(open(filename).read().split('\n'))
    except UnicodeDecodeError:
        size = 0
    files[i] = { 'path': files[i], 'size': size }

# get a list of largest files
files.sort(key=lambda f: (-f['size'], f['path']))
print('lines', 'file')
for file in files[:20]:
    if file['size'] == 0:
        break
    print(' '.join([str(file['size']), file['path']]))
total = sum([file['size'] for file in files])
print("Total", total, "lines.")

