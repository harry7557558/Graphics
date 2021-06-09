# generate colors.min.js from three js files

import re

poly = open("poly.js", "r").read().strip()
trig = open("trig.js", "r").read().strip()
cosine = open("cosine.js", "r").read().strip()

js = "const colorFunctions = {" \
    + poly.replace("const ColorFunctions =", "poly:").rstrip(';') + ',' \
    + trig.replace("const ColorFunctions =", "trig:").rstrip(';') + ',' \
    + cosine.replace("const ColorFunctions =", "cosine:").rstrip(';') \
    + "};"

js = re.sub('\s+', ' ', js)
js = re.sub(' *([\,\;\:\{\}\(\)\=]) ', r'\1', js)

open("colors.min.js", "w").write(js)
