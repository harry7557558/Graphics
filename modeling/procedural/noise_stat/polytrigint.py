# A file created to be used instead of sympy.integrate
# Expected to be faster than SymPy

from fractions import Fraction
from copy import deepcopy


def split_str_bracket(s: str, sep: str, clear_bracket: bool):
    """
        Split @s by @sep without breaking expressions inside the bracket
    """
    if len(sep) != 1:
        raise ValueError("separator must be one character in length")
    if sep in "()[]{}":
        raise ValueError("separator cannot be bracket")
    s += sep

    res = []
    bracket_nest_count = 0
    t = ""
    has_zero = True
    for c in s:
        if c in "([{":
            if bracket_nest_count == 0 and len(t) > 0 and t[-1] in ")]}":
                has_zero = True
            bracket_nest_count += 1
        elif c in ")]}":
            bracket_nest_count -= 1
            if bracket_nest_count <= 0:
                bracket_nest_count = 0
        elif bracket_nest_count == 0 and c != sep:
            has_zero = True
        if c == sep and bracket_nest_count == 0:
            if clear_bracket and not has_zero:
                while len(t) >= 2 and (t[0] in "([{") and (t[-1] in ")]}"):
                    t = t[1:len(t)-1]
            res.append(t)
            t = ""
            has_zero = False
        else:
            t += c
    if bracket_nest_count != 0:
        res.append(t[0:len(t)-1])
    return res


class Constant():

    """
        Sum of terms in the form a/b*pi^n and a/b*sin(pi*u+v)
    """

    def __init__(self, s=1):
        if type(s) in [int, float, Fraction]:
            self.f = Fraction(s)
        elif type(s) == Constant:
            self.f = s.f
        elif type(s) == str:
            self.f = Fraction(s)
        else:
            raise TypeError("Unsupported Constant type: " + str(type(s)))

    def __str__(self):
        return str(self.f)

    def __eq__(self, s):
        s = Constant(s)
        return self.f == s.f

    def __add__(a, b):
        c = Constant()
        c.f = a.f+b.f
        return c

    def __mul__(a, b):
        c = Constant()
        c.f = a.f * b.f
        return c

    def __neg__(a):
        c = Constant()
        c.f = -a.f
        return c

    def __sub__(a, b):
        c = Constant()
        c.f = a.f - b.f
        return c

    def __div__(a, b):
        c = Constant()
        c.f = a.f / b.f
        return c

    def __pow__(a, b:int):
        c = Constant
        c.f = a**b
        return c


class TrigMonomial():

    """
        coe: Constant
        symbols: { 'x': [2, [3], [4, 5]] } means x^2*sin(3*x)*cos(4*x)*cos(5*x)
    """

    def __init__(self, s=1):
        self.coe = Constant(1)
        self.symbols = {}

        if type(s) in [int, float, Fraction]:
            self.coe *= Constant(s)
            return
        elif type(s) == TrigMonomial:
            self.coe = deepcopy(s.coe)
            self.symbols = deepcopy(s.symbols)
            return
        elif type(s) != str:
            raise TypeError("Unsupported TrigMonomial type: " + str(type(s)))

        # No guarantee to work for complex/unconventional strings

        s = s.replace('**', '^')
        s = s.replace('[', '(').replace(']', ')')
        s = s.replace('{', '(').replace('}', ')')
        s = split_str_bracket(s, '*', True)

        for t in s:
            if len(t) == 0:
                continue
            try:  # number
                k = Constant(t)
                self.coe *= k
            except:  # string
                if t[0] == '-':
                    self.coe = -self.coe
                    t = t[1:]
                if '^' in t:
                    t, p = t.split('^')
                    p = int(p)
                    if p < 0:
                        raise ValueError("Exponent must be non-negative")
                else:
                    p = 1
                # variable name
                if t.isalnum():
                    if t in self.symbols:
                        self.symbols[t][0] += p
                    else:
                        self.symbols[t] = [p, [], []]
                # trigonometric function
                elif len(t) > 4 and t[0:4] in ["sin(", "cos("] and t[-1] == ')':
                    index = 1 if t[0:4] == "sin(" else 2
                    t = t[4:len(t)-1]
                    if '*' in t:  # sin(3*x), cos(6*x)
                        if t.count('*') != 1:
                            raise NotImplementedError("Multiple multiplication signs inside trigonometric")
                        u, v = t.split('*')
                        if u.lstrip('-').isnumeric() and v.isalnum():
                            k, x = int(u), v
                        elif u.isalnum() and v.lstrip('-').isnumeric():
                            k, x = int(v), u
                        else:
                            raise ValueError("Expect one integer and one variable name")
                        if not x in self.symbols:
                            self.symbols[x] = [0, [], []]
                        self.symbols[x][index] += [k]*p
                    else:  # cos(x), sin(-x)
                        k = -1 if t[0] == '-' else 1
                        t = t.lstrip('-')
                        if t.isnumeric():
                            raise NotImplementedError("Unsupported number inside trigonometric")
                        if not t.isalnum():
                            raise ValueError(f"Expect a variable name ({t})")
                        if not t in self.symbols:
                            self.symbols[t] = [0, [], []]
                        self.symbols[t][index] += [k]*p
                else:
                    raise ValueError(f"Unrecognized name {t}")

    def __str__(self):
        res = [str(self.coe)]
        for s in self.symbols:
            p = self.symbols[s][0]
            if p != 0:
                res.append(s if p == 1 else f"{s}**{self.symbols[s][0]}")
            for c in self.symbols[s][1]:
                res.append(f"sin({c}*{s})")
            for c in self.symbols[s][2]:
                res.append(f"cos({c}*{s})")
        return '*'.join(res)

    def __mul__(a, b):
        c = TrigMonomial()
        c.coe = a.coe * b.coe
        for s in a.symbols:
            v = a.symbols[s]
            for t in b.symbols:
                if s == t:
                    w = b.symbols
                    v[0] += w[0]
                    v[1] += w[1]
                    v[2] += w[2]
            c.symbols[s] = v
        for s in b.symbols:
            if s not in a.symbols:
                c.symbols[s] = deepcopy(b.symbols[s])
        return c

    def simplify(self):
        res = TrigMonomial()
        res.coe = self.coe
        sign = 1
        for varname in self.symbols:
            var = deepcopy(self.symbols[varname])
            for i in range(len(var[1])):  # sin
                if var[1][i] == 0:
                    sign = 0
                    var[1] = []
                    break
                elif var[1][i] < 0:
                    var[1][i] *= -1
                    sign *= -1
            var[1] = sorted(var[1])
            for i in range(len(var[2])):  # cos
                if var[2][i] < 0:
                    var[2][i] *= -1
            var[2] = sorted(var[2])
            while len(var[2]) > 0 and var[2][0] == 0:
                var[2] = var[2][1:]
            if var != [0, [], []]:
                res.symbols[varname] = var
        res.coe *= Constant(sign)
        if res.coe == 0:
            res.symbols = {}
        return res


class TrigPolynomial():

    def __init__(self, s=0):
        self.terms = []
        
        if type(s) == TrigPolynomial:
            self.terms = deepcopy(s.terms)
            return
        elif type(s) in [int, float, Fraction, TrigMonomial]:
            if s != 0:
                self.terms.append(TrigMonomial(s))
            return
        elif type(s) in [list, tuple]:
            self.terms = [TrigMonomial(t) for t in s]
        elif type(s) != str:
            raise TypeError("Unsupported TrigMonomial type: " + str(type(s)))

        # just lazy
        s = '('+s+')'
        s = s.replace('-', '+-').replace('(+', '(').replace('++', '+')
        s = split_str_bracket(s[1:len(s)-1], '+', True)
        self.terms = [TrigMonomial(t) for t in s]

    def __str__(self):
        if len(self.terms) == 0:
            return '0'
        s = '+'.join([str(si) for si in self.terms])
        return s.replace('+-', '-')

    def simplify(self):
        """ intended to reduce the number of terms to increase speed """
        p = self
        q = TrigPolynomial()
        for t in p.terms:
            t = t.simplify()
            if t.coe != 0:
                q.terms.append(t)
        return q

    def __add__(p, q):
        if type(q) != TrigPolynomial:
            q = TrigPolynomial(q)
        r = TrigPolynomial()
        r.terms = p.terms + q.terms
        return r.simplify()

    def __neg__(p):
        q = deepcopy(p)
        for i in range(len(q.terms)):
            q.terms[i].coe *= -1
        return q

    def __sub__(p, q):
        if type(q) != TrigPolynomial:
            q = TrigPolynomial(q)
        return p.__add__(q.__neg__())

    def __mul__(p, q):
        if type(q) != TrigPolynomial:
            q = TrigPolynomial(q)
        r = TrigPolynomial()
        for pi in p.terms:
            for qi in q.terms:
                r.terms.append(pi*qi)
        return r.simplify()

    def __pow__(p, e: int):
        r = TrigPolynomial(1)
        x = deepcopy(p)
        while e:
            if e & 1:
                r = r * x
            e >>= 1
            if e:
                x = x * x
        return r

    def __radd__(p, q):
        return p.__add__(q)

    def __rmul__(p, q):
        return TrigPolynomial(q).__mul__(p)


def fourier_series(c:list, s:list):
    """
        Not really.
        Rewrite: prod(cos(c*x))*prod(sin(s*x))
        To: sum(rc*cos(i*x))+sum(rs*sin(i*x))
        Return ac and as.
        I believe this function has an O(N^2) time complexity.
    """
    for _ in c+s:
        if not (type(_) == int and _ > 0):
            raise ValueError("Harmonic must be positive integer")

    is_odd = len(s)%2 == 1  # odd or even function
    deg = sum(c)+sum(s)

    # the weights of every cos "harmonic"
    r = [Fraction(1)]+[Fraction(0)]*deg  # 1*cos(0*x)

    # multiply by each cos(k*x) term
    for k1 in c:
        r0, r = r, [Fraction(0)]*(deg+1)
        # cos(x)*cos(y) = 1/2*cos(x-y) + 1/2*cos(x+y)
        for k2 in range(len(r0)):
            if r0[k2] != 0:
                r[abs(k1-k2)] += r0[k2]*Fraction(1, 2)
                r[k1+k2] += r0[k2]*Fraction(1, 2)

    # multiply by each sin(k*x) term
    for i in range(0, len(s)-int(is_odd), 2):
        f1, f2 = s[i], s[i+1]
        # sin(x)*sin(y) = 1/2*cos(x-y) + -1/2*cos(x+y)
        k11, k12 = abs(f1-f2), f1+f2
        r0, r = r, [Fraction(0)]*(deg+1)
        for k2 in range(len(r0)):
            if r0[k2] != 0:
                r[abs(k11-k2)] += r0[k2]*Fraction(1, 4)
                r[k11+k2] += r0[k2]*Fraction(1, 4)
                r[abs(k12-k2)] -= r0[k2]*Fraction(1, 4)
                r[k12+k2] -= r0[k2]*Fraction(1, 4)

    if is_odd:
        # multiply by the missed sine term, all terms become sine
        k1 = s[-1]
        r0, r = r, [Fraction(0)]*(deg+1)
        # sin(x)*cos(y) = 1/2*sin(x+y) + 1/2*sin(x-y)
        for k2 in range(len(r0)):
            if r0[k2] != 0:
                r[k1+k2] += r0[k2]*Fraction(1, 2)
                sign = -1 if k1 < k2 else 1
                r[abs(k1-k2)] += r0[k2]*Fraction(sign, 2)
        r[0] = Fraction(0)
        return [], r
    else:
        # all terms are cosine
        return r, []


def integrate_polycos(p:int, k:int):
    """
        \int x^p \cos(kx) dx
        return two polynomial lists (pc, ps); pc(x)*cos(k*x)+ps(x)*sin(k*x)
    """
    if p == 0:
        return ([0], [Fraction(1, k)])
    if k == 0:
        pc = [Fraction(0)]*(p+1) + [Fraction(1, p+1)]
        return (pc, [0])

    # \int x^p\cos(kx)dx = \frac{1}{k} x^p\sin(kx) - \frac{p}{k} \int x^{p-1}\sin(kx)dx
    pc, ps = [Fraction(0)]*(p+1), [Fraction(0)]*(p+1)
    ps[p] += Fraction(1, k)
    rc, rs = integrate_polysin(p-1, k)
    for i in range(p):
        pc[i] -= Fraction(p, k) * rc[i]
        ps[i] -= Fraction(p, k) * rs[i]
    return (pc, ps)
        

def integrate_polysin(p:int, k:int):
    """
        \int x^p \sin(kx) dx
        return two polynomial lists (pc, ps); pc(x)*cos(k*x)+ps(x)*sin(k*x)
    """
    if p == 0:
        return ([Fraction(-1, k)], [0])
    if k == 0:
        return ([0], [0])

    # \int x^p\sin(kx)dx = -\frac{1}{k} x^p\cos(kx) + \frac{p}{k} \int x^{p-1}\cos(kx)dx
    pc, ps = [Fraction(0)]*(p+1), [Fraction(0)]*(p+1)
    pc[p] -= Fraction(1, k)
    rc, rs = integrate_polycos(p-1, k)
    for i in range(p):
        pc[i] += Fraction(p, k) * rc[i]
        ps[i] += Fraction(p, k) * rs[i]
    return (pc, ps)


def integrate_monomial(varname:str, symbol:list):
    """
        Receive something like ('x', [3, [1], [2]]) representing x**3*sin(x)*cos(2*x)
        Return a TrigPolynomial, its indefinite integral
    """
    p = symbol[0]
    ans = TrigPolynomial()
    if symbol[1] == [] and symbol[2] == []:
        a = Monomial()
        a.coe = Constant(Fraction(1, p+1))
        a.symbols[varname] = [p+1, [], []]
        ans.terms.append(deepcopy(a))
        return ans

    fc, fs = fourier_series(symbol[2], symbol[1])
    if len(fc) > 0:
        f = fc
        integrate = integrate_polycos
    if len(fs) > 0:
        f = fs
        integrate = integrate_polysin
    for k in range(len(f)):
        if f[k] == 0:
            continue
        a = TrigMonomial()
        rc, rs = integrate(p, k)
        for i in range(len(rc)):
            a.coe = Constant(rc[i]*f[k])
            a.symbols[varname] = [i, [], [k]]
            ans.terms.append(deepcopy(a))
        for i in range(len(rs)):
            a.coe = Constant(rs[i]*f[k])
            a.symbols[varname] = [i, [k], []]
            ans.terms.append(deepcopy(a))
    return ans.simplify()
    

def integrate(poly, args):
    if type(poly) == TrigMonomial:
        if len(args) == 0:
            return poly
        a = args[0]
        ans = TrigPolynomial()
        if type(a) == str:
            m = deepcopy(poly)
            if a in poly.symbols:
                m.symbols.pop(a)
                p = integrate_monomial(a, poly.symbols[a])
                ans = p * m
            else:
                m.symbols[a] = [1, [], []]
                ans.terms.append(m)
        elif type(a) == tuple:
            raise NotImplementedError()
        else:
            raise TypeError('Integral argument must be variable name or tuple')
        return integrate(ans, args[1:])

    elif type(poly) == TrigPolynomial:
        p = deepcopy(poly).simplify()
        ans = TrigPolynomial()
        for t in p.terms:
            ans += integrate(t, args)
        return ans
    else:
        return integrate(TrigPolynomial(poly), args)
    


# testing

if __name__ == "__main__":

    s = "x^2*sin(x)*cos(2*x)+cos(3*x)*sin(-x)^2+sin(7*x)*x^5"
    p = TrigPolynomial(s)
    print(p)
    i = integrate(s, 'x')
    print(i)
