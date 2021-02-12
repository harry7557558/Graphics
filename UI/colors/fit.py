# get color function from color images

from PIL import Image
import numpy as np

def fitImage(img):
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
    x = np.array([(i+.5)/w for i in range(w)])
    vals = np.array(vals).transpose()
    coes = []
    for v in vals:
        for deg in range(3,11):
            coes_t = np.polyfit(x,v,deg, full=True)
            loss = coes_t[1][0]
            if loss<0.01:
                break
        print("deg={};".format(deg), "loss = {:.6f}".format(coes_t[1][0]))
        coes.append(list(coes_t[0]))
    return coes


names = open("img/index","r").read().split('\n')


JS = """const ColorFunctions = {
\tclp: function(x) {
\t\treturn Math.floor(255.99*(x<0.?0.:x>1.?1.:x));
\t},
"""

CPP = """template<typename vec3, typename Float>
class ColorFunctions {
\tstatic Float clp(double x) {
\t\treturn (Float)(x<0.?0.:x>1.?1.:x);
\t}
public:
"""

for i in range(0,50):
    img = Image.open("img/"+str(i+1)+".png")
    print(names[i])
    coes = fitImage(img)
    JS += "\t" + names[i] + ": function(t) {\n"
    CPP += "\tstatic vec3 " + names[i] + "(double t) {\n"
    for c in range(3):
        deg = len(coes[c])-1
        st = '('*(deg-1) + ')*t'.join(["{:+.6f}".format(coes[c][i]) for i in range(deg+1)]).lstrip('+').replace(')','',1)
        JS += "\t\tvar " + 'rgb'[c] + " = " + st + ';\n'
        CPP += "\t\tdouble " + 'rgb'[c] + " = " + st + ';\n'   # double precision required
    JS += "\t\treturn 'rgb('+this.clp(r)+','+this.clp(g)+','+this.clp(b)+')';\n\t},\n"
    CPP += "\t\treturn vec3(clp(r),clp(g),clp(b));\n\t}\n"

JS += "};"
CPP += "};\n"

open("ColorFunctions.js","wb").write(bytearray(JS,'utf-8'))
open("ColorFunctions.h","wb").write(bytearray(CPP,'utf-8'))


