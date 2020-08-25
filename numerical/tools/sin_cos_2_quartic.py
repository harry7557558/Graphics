# Approximate trigonometric functions using quartic polynomials
# Find the coefficients of the quartic polynomial

# Multivation:
# Need to solve equations involving trigonometric functions
# Quartic equations can be solved analytically
# After finding the approximations of roots, perform Newton's iteration to find numerical solution

# Arbitrary precision floating point
from decimal import *
getcontext().prec=40

# 100 decimal digits of π
pi = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')

# Analytical coefficients of polynomial of best fit in [0,pi/2]
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
sp = [0, sp1, sp2, sp3, sp4]
cp = [1, cp1, cp2, cp3, cp4]

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

# reflect a polynomial about x axis
def reflectY(p):
    return [-t for t in p]

# Split [0,2π] into 4 intervals
SP = [
    sp,
    translate(cp,pi/2),
    translate(reflectY(sp),pi),
    translate(reflectY(cp),3*pi/2) ]
CP = [
    cp,
    translate(SP[2],-pi/2),
    translate(SP[3],-pi/2),
    translate(sp,3*pi/2) ]

# Output for Desmos graphing calculator
Intervals = [
    '\\left\\{0<x<\\frac{\\pi}{2}\\right\\}',
    '\\left\\{\\frac{\\pi}{2}<x<\\pi\\right\\}',
    '\\left\\{\\pi<x<\\frac{3\\pi}{2}\\right\\}',
    '\\left\\{\\frac{3\\pi}{2}<x<2\\pi\\right\\}' ]
print('Sin[x]')
for i in range(4):
    printPolynomial(SP[i], Intervals[i]+'\n')
print('Cos[x]')
for i in range(4):
    printPolynomial(CP[i], Intervals[i]+'\n')
print(end='\n')

# Output for C++
print('const double SP[4][5] = {')
for i in range(4):
    printArray(SP[i])
print('}, CP[4][5] = {')
for i in range(4):
    printArray(CP[i])
print('};')

