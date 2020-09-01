# Python high precision numerical works
# Still fails in degenerated cases (avoid it)

# Fitting polynomial to trigonometric functions:
# Given degree N, interval [a, b], find the coefficients of the polynomial of best fit

from decimal import *
getcontext().prec=80

pi = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')

# https://docs.python.org/3/library/decimal.html#decimal-recipes
def sin(x):
    getcontext().prec += 2
    i, lasts, s, fact, num, sign = 1, 0, x, 1, x, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i-1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    getcontext().prec -= 2
    return +s
def cos(x):
    getcontext().prec += 2
    i, lasts, s, fact, num, sign = 0, 0, 1, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i-1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    getcontext().prec -= 2
    return +s


# Integral[x^n*sin(x),a,b]
# Integral[x^n*cos(x),a,b]
def SinInt(n, a, b):
    if n==0: return -cos(b)+cos(a)
    return -(b**n)*cos(b)+(a**n)*cos(a) + n*CosInt(n-1,a,b)
def CosInt(n, a, b):
    if n==0: return sin(b)-sin(a)
    return (b**n)*sin(b)-(a**n)*sin(a) - n*SinInt(n-1,a,b)

# Integral[(1,x,x^2...x^(n-1))*(1,x,x^2...x^(n-1))^T,a,b]
def PolyCovarianceInt(n, a, b):
    M = [[0 for u in range(n)] for v in range(n)]
    for i in range(n):
        for j in range(n):
            e = i+j+1
            M[i][j] = (b**e-a**e)/e
    return M

# Integral[(1,x,x^2...x^(n-1))*sin(x),a,b]
# Integral[(1,x,x^2...x^(n-1))*cos(x),a,b]
def PolySinInt(n, a, b):
    return [SinInt(n, a, b) for n in range(n)]
def PolyCosInt(n, a, b):
    return [CosInt(n, a, b) for n in range(n)]

# Gaussian elimination
def solveLinear(A, x):
    N = len(x)
    for i in range(N):
        Ai = A[i]
        m = 1 / Ai[i]
        for k in range(i,N):
            Ai[k] *= m
        x[i] *= m
        for j in range(N):
            if j!=i:
                Aj = A[j]
                m = -Aj[i]/Ai[i]
                for k in range(i,N):
                    Aj[k] += m*Ai[k]
                x[j] += m*x[i]

# Polynomials of best fit
def fitSin(N, a, b):
    N+=1
    M = PolyCovarianceInt(N, a, b)
    x = PolySinInt(N, a, b)
    solveLinear(M, x)
    return x
def fitCos(N, a, b):
    N+=1
    M = PolyCovarianceInt(N, a, b)
    x = PolyCosInt(N, a, b)
    solveLinear(M, x)
    return x

# Polynomials of best fit with boundary constraints
def fitTrigonometric_c(F, f, N, a, b):
    N+=1
    M = PolyCovarianceInt(N, a, b)
    x = F(N, a, b)
    l1, l2 = [], []
    for i in range(N):
        ai = Decimal(a**i if i else 1)/2
        bi = Decimal(b**i if i else 1)/2
        M[i] += [ai,bi]
        l1.append(ai)
        l2.append(bi)
    M += [l1+[0,0],l2+[0,0]]
    x += [f(a)/2,f(b)/2]
    solveLinear(M, x)
    return x[:N]
def fitSin_c(N, a, b):
    return fitTrigonometric_c(PolySinInt, sin, N, a, b)
def fitCos_c(N, a, b):
    return fitTrigonometric_c(PolyCosInt, cos, N, a, b)

# print polynomial coefficients for graphing calculators
def printPolynomial(arr, end='\n'):
    s = ''
    for i in range(len(arr)):
        v = arr[i]
        if v*v>Decimal('1e-40'):
            s += ('+' if v>0 else '-') + str(v if v>0 else -v)[:16]
            s += ('*x^{'+str(i)+'}') if i>1 else ('' if i==0 else '*x')
    print(s.lstrip('+'), end=end)

# print coefficients as C++ style initializer list
def printArray(arr, end=',\n'):
    s = '{ '
    for i in arr:
        s += str(i.quantize(Decimal('1E-20'))) + ', '
    s = s.rstrip(', ').replace('0E-20','0.'+'0'*20) + ' }'
    print(s, end=end)



# Print result; Split [0,2π] to N equal intervals and fit to quadratic functions
# N should be even
def QuadraticCoefficients(fitTrig, N):
    # calculate coefficients
    dt = 2*pi/N
    C = [fitTrig(2, i*dt, (i+1)*dt) for i in range(N)]
    # graphing calculator output
    print('C(x)=\\left\\{',end='')
    for i in range(N):
        if i+1!=N: print('x<\\frac{'+str(i+1)+'\\pi}{'+str(N//2)+'}',end=':')
        printPolynomial(C[i], '' if i+1==N else ',')
    print('\\right\\}\n')
    # C++ initializer list output with shifting
    print('const double C['+str(N)+'][3] = {', end=' ')
    for i in range(N):
        d = i*pi/(N//2)
        c = C[i]
        printArray([c[0]+c[1]*d+c[2]*d*d,c[1]+2*c[2]*d,c[2]], ', ')
    print('};')

QuadraticCoefficients(fitCos_c, 32)




