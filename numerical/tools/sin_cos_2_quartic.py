# Approximate trigonometric functions using quartic polynomials
# Find the coefficients of the quartic polynomial

# Motivation:
# Need to solve equations involving trigonometric functions
# Quartic equations can be solved analytically
# After finding the approximations of roots, perform Newton's iteration to find a numerical solution

# Arbitrary precision floating point
from decimal import *
getcontext().prec=40

# 100 decimal digits of π and √2
pi = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')
rt2 = Decimal('1.414213562373095048801688724209698078569671875376948073176679737990732478462107038850387534327641573')

# Analytical coefficients of the polynomial of best fit in [0,pi/2]
# Thanks Wolfram open access
sp1 = -5*(1161216-274176*pi-33600*pi**2+1008*pi**3+pi**5)/pi**6
sp2 = 70*(940032-228096*pi-25536*pi**2+912*pi**3+pi**5)/pi**7
sp3 = -280*(774144-191232*pi-20160*pi**2+816*pi**3+pi**5)/pi**8
sp4 = (2/pi)**4-(2/pi)**3*sp1-(2/pi)**2*sp2-(2/pi)*sp3
cp1 = -(5806080-1532160*pi-127680*pi**2+8400*pi**3+23*pi**5)/pi**6
cp2 = 14*(4700160-1209600*pi-110400*pi**2+6000*pi**3+11*pi**5)/pi**7
cp3 = -56*(3870720-979200*pi-95040*pi**2+4560*pi**3+7*pi**5)/pi**8
cp4 = -(2/pi)**4-(2/pi)**3*cp1-(2/pi)**2*cp2-(2/pi)*cp3
# lowest coefficient at first
sp_4 = [0, sp1, sp2, sp3, sp4]
cp_4 = [1, cp1, cp2, cp3, cp4]

# Polynomial of best fit in [0,pi/4]
sp1 = -5*(74317824-37158912*rt2-4386816*rt2*pi-537600*pi**2+204288*rt2*pi**2+4032*rt2*pi**3+rt2*pi**5)/pi**6
sp2 = 140*(60162048-30081024*rt2-3649536*rt2*pi-408576*pi**2+176640*rt2*pi**2+3648*rt2*pi**3+rt2*pi**5)/pi**7
sp3 = -1120*(49545216-24772608*rt2-3059712*rt2*pi-322560*pi**2+152064*rt2*pi**2+3264*rt2*pi**3+rt2*pi**5)/pi**8
sp4 = (256/(pi**4))*(rt2/2-pi/4*sp1-pi**2/16*sp2-pi**3/64*sp3)
cp1 = -(185794560*rt2-49029120*pi-21934080*rt2*pi-1021440*rt2*pi**2+67200*pi**3+20160*rt2*pi**3+46*pi**5+5*rt2*pi**5)/pi**6
cp2 = 28*(150405120*rt2-38707200*pi-18247680*rt2*pi-883200*rt2*pi**2+48000*pi**3+18240*rt2*pi**3+22*pi**5+5*rt2*pi**5)/pi**7
cp3 = -224*(123863040*rt2-31334400*pi-15298560*rt2*pi-760320*rt2*pi**2+36480*pi**3+16320*rt2*pi**3+14*pi**5+5*rt2*pi**5)/pi**8
cp4 = (256/(pi**4))*(rt2/2-1-pi/4*cp1-pi**2/16*cp2-pi**3/64*cp3)
sp_8 = [0, sp1, sp2, sp3, sp4]
cp_8 = [1, cp1, cp2, cp3, cp4]


# print coefficients as C++ style initializer list
def printArray(arr):
    s = '{ '
    for i in arr:
        s += str(i)[:24] + ', '
    s = s.rstrip(', ') + ' },'
    print(s)

# print as polynomials for graphing calculators
def printPolynomial(arr, end='\n'):
    s = ''
    for i in range(len(arr)):
        v = arr[i]
        if v!=Decimal(0):
            s += ('+' if v>0 else '-') + str(v if v>0 else -v)[:16]
            s += ('*x^'+str(i)) if i>1 else ('' if i==0 else '*x')
    print(s.lstrip('+'), end=end)


# translate a quartic polynomial right by d
def translate(p, d):
    q = []
    q.append(p[0]-p[1]*d+p[2]*d**2-p[3]*d**3+p[4]*d**4)
    q.append(p[1]-2*p[2]*d+3*p[3]*d**2-4*p[4]*d**3)
    q.append(p[2]-3*p[3]*d+6*p[4]*d**2)
    q.append(p[3]-4*p[4]*d)
    q.append(p[4])
    return q

# reflect a polynomial about the x-axis
def reflectY(p):
    return [-t for t in p]

# reflect a polynomial about the y-axis
def reflectX(p):
    q = []
    for i in range(len(p)):
        q.append(-p[i] if i&1 else p[i])
    return q

# Split [0,2π] into 4 and 8 intervals
SP_4 = [
    sp_4,
    cp_4,
    reflectY(sp_4),
    reflectY(cp_4) ]
SP_8 = [
    sp_8,
    translate(reflectX(cp_8),pi/4),
    cp_8,
    translate(reflectX(sp_8),pi/4) ]
SP_8 += [
    reflectY(SP_8[0]),
    reflectY(SP_8[1]),
    reflectY(SP_8[2]),
    reflectY(SP_8[3]) ]

# Output for Desmos graphing calculator
print('SP_4[x]')
for i in range(4):
    printPolynomial(SP_4[i], '\\left\\{0<x<\\frac{\\pi}{2}\\right\\}\n')
print('SP_8[x]')
for i in range(8):
    printPolynomial(SP_8[i], '\\left\\{0<x<\\frac{\\pi}{4}\\right\\}\n')
print(end='\n')

# Output for C++
print('const double SP_4[4][5] = {')
for i in range(4):
    printArray(SP_4[i])
print('};')
print('const double SP_8[8][5] = {')
for i in range(8):
    printArray(SP_8[i])
print('};')

