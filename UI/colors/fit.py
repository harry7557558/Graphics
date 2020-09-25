# get color function from color images

from PIL import Image
import numpy as np

def fitImage(img, deg):
    # load pixels
    img = img.crop(img.getbbox())
    w, h = img.size
    pixels = img.load()
    # take color the average on vertical direction
    vals = []
    for i in range(w):
        s = (0,0,0)
        for j in range(h):
            c = pixels[i,j]
            s = (s[0]+c[0],s[1]+c[1],s[2]+c[2])
        vals.append((s[0]/(256.*h),s[1]/(256.*h),s[2]/(256.*h)))
    # polynomial fitting
    x = [(i+.5)/w for i in range(w)]
    coes = np.polyfit(x,vals,deg)
    return coes


names = open("img/index","r").read().split('\n')
deg = 6


JS = """const ColorFunctions = {
\tclp: function(x) {
\t\treturn Math.floor(255.99*(x<0.?0.:x>1.?1.:x));
\t},
"""

CPP = """// You need a vec3 class to use this
#pragma once
class ColorFunctions {
\tstatic double clp(double x) {
\t\treturn x<0.?0.:x>1.?1.:x;
\t}
public:
"""

for i in range(0,50):
    img = Image.open("img/"+str(i+1)+".png")
    coes = fitImage(img, deg)
    JS += "\t" + names[i] + ": function(t) {\n"
    CPP += "\tstatic vec3 " + names[i] + "(double t) {\n"
    for c in range(3):
        st = 'rgb'[c] + " = " + '('*(deg-1) + ')*t'.join(["{:+.6f}".format(coes[i][c]) for i in range(deg+1)]).lstrip('+').replace(')','',1)
        JS += "\t\tvar " + st + ';\n'
        CPP += "\t\tdouble " + st + ';\n'
    JS += "\t\treturn 'rgb('+this.clp(r)+','+this.clp(g)+','+this.clp(b)+')';\n\t},\n"
    CPP += "\t\treturn vec3(clp(r),clp(g),clp(b));\n\t}\n"

JS += "};"
CPP += "};\n"

open("ColorFunctions.js","wb").write(bytearray(JS,'utf-8'))
open("ColorFunctions.h","wb").write(bytearray(CPP,'utf-8'))


