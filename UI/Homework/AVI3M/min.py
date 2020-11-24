# batch image compressor

import os
_dir = os.path.dirname(__file__)+'\\'
files = [f for f in os.listdir(_dir) if f.find('.png')!=-1]

from PIL import Image
for filename in files:
    img = Image.open(_dir+filename).convert('RGB')
    img = img.resize((img.size[0]//2,img.size[1]//2), Image.ANTIALIAS)
    img.save(_dir+'min\\'+filename.replace('.png','.jpg'), 'JPEG', quality=90)

