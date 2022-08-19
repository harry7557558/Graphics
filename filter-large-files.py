# Filter large files

import subprocess
import os

# get a list of files
files = subprocess.run(
    ['git', 'ls-tree', '--full-tree', '--name-only', '-r', 'master'],
    stdout=subprocess.PIPE
)
files = files.stdout.decode('utf-8').strip().split('\n')

# get file sizes
for i in range(len(files)):
    filename = files[i]
    if os.path.isfile(filename):
        size = os.stat(files[i]).st_size
    else:
        size = 0
    size *= int(filename.startswith('UI'))
    files[i] = { 'path': filename, 'size': size }

# get a list of largest files
files.sort(key=lambda f: -f['size'])
for file in files[:20]:
    print(' '.join([str(file['size']), file['path']]))
total = sum([file['size'] for file in files])
print("Total", total, "bytes.")
