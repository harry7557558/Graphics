# A file created to be used instead of sympy.integrate
# Expected to be faster than SymPy

# Calculate the integral of the product of polynomials and trigometric series

from fractions import Fraction
from copy import deepcopy
import re
import math


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


class PolyPi():

    """ sum(c*pi^p), pows[p]=c """

    def __init__(self, s={}):
        self.pows = {}
        if type(s) == PolyPi:
            self.pows = deepcopy(s.pows)
        elif type(s) == dict:
            self.pows = deepcopy(s)
        elif type(s) == str:
            s = s.replace('**', '^')
            s = re.sub(r"pi([^\^])", r"pi^1\g<1>", s+' ').replace(' ', '')
            s = s.replace('^-', '^_')
            s = s.replace('-', '+-').replace('++', '+')
            s = s.replace('_', '-')
            s = s.split('+')
            for t in s:
                if not '*' in t:
                    if 'pi' in t:
                        f = Fraction(1)
                        p = int(t.split('^')[1])
                    else:
                        f = Fraction(t)
                        p = 0
                else:
                    f, p = t.split('*')
                    f = Fraction(f)
                    p = int(p.split('^')[1])
                if p not in self.pows:
                    self.pows[p] = Fraction(0)
                self.pows[p] += f
        else:
            s = Fraction(s)
            if s != 0:
                self.pows[0] = s

    def __str__(self):
        ts = []
        for p in self.pows:
            str_pp = str(self.pows[p])
            if str_pp == '1' and p != 0:
                str_pp = ''
            if p == 0:
                s = str_pp
            elif p == 1:
                s = str_pp + "*pi"
            else:
                s = str_pp + "*pi**" + str(p)
            ts.append(s.lstrip('*'))
        if len(ts) == 0:
            return '0'
        if len(ts) == 1:
            return ts[0]
        return '(' + '+'.join(ts) + ')'

    def simplify(self):
        res = PolyPi()
        for p in self.pows:
            if self.pows[p] != 0:
                res.pows[p] = self.pows[p]
        return res

    def __eq__(a, b):
        if type(b) in [Constant, TrigMonomial, TrigPolynomial]:
            return b.__eq__(a)
        if type(b) not in [int, float, Fraction, PolyPi]:
            return False
        a = a.simplify()
        b = PolyPi(b)
        return a.pows == b.pows

    def __add__(a, b):
        c = deepcopy(a)
        b = PolyPi(b)
        for p in b.pows:
            if p not in c.pows:
                c.pows[p] = Fraction(0)
            c.pows[p] += b.pows[p]
        return c

    def __neg__(a):
        c = deepcopy(a)
        for p in c.pows:
            c.pows[p] *= -1
        return c

    def __sub__(a, b):
        return a.__add__(b.__neg__())

    def __mul__(a, b):
        b = PolyPi(b)
        c = PolyPi()
        for pa in a.pows:
            for pb in b.pows:
                p = pa + pb
                val = a.pows[pa] * b.pows[pb]
                if p not in c.pows:
                    c.pows[p] = val
                else:
                    c.pows[p] += val
        return c

    def __truediv__(a, b):
        if type(b) == PolyPi:
            if len(b.pows) != 1:
                raise ValueError("Dividing by PolyPi is not supported.")
            for pb in b.pows:
                m = 1 / b.pows[pb]
                c = PolyPi()
                for pa in a.pows:
                    p = pa - pb
                    val = m * a.pows[pa]
                    c.pows[p] = val
            return c
        else:
            b = 1 / Fraction(b)
            c = deepcopy(a)
            for p in c.pows:
                c.pows[p] *= b
            return c

    def __radd__(p, q):
        return p.__add__(q)

    def __rmul__(p, q):
        return p.__mul__(q)

    def __float__(self):
        """ Numerical evaluation """
        ans = 0.0
        for p in self.pows:
            v = float(self.pows[p])
            ans += v * math.pi**p
        return ans

    def normalize_0_2pi(self):
        """ Normalize the pi^1 term to [0,2) """
        q = deepcopy(self)
        if 1 in self.pows:
            q.pows[1] = self.pows[1] % 2
        return q

    def __abs__(self):
        if float(self) >= 0.0:
            return self
        else:
            return self.__neg__()

    def is_integer(self):
        if len(self.pows) == 0:
            return True
        if len(self.pows) > 1:
            return False
        return (0 in self.pows) and (self.pows[0].denominator == 1)

    def __int__(self):
        if 0 not in self.pows:
            return 0
        return self.pows[0].numerator // self.pows[0].denominator


PI_12 = PolyPi('1/2*pi')
PI_22 = PolyPi('2/2*pi')
PI_32 = PolyPi('3/2*pi')


class Constant():

    """
        f + sm * cos(sk)
    """

    def __init__(self, s=0):
        self.f = PolyPi()
        self.cm = []
        self.ck = []
        if type(s) in [int, float, Fraction, PolyPi]:
            self.f = PolyPi(s)
            return
        elif type(s) == Constant:
            self.f = deepcopy(s.f)
            self.cm = deepcopy(s.cm)
            self.ck = deepcopy(s.ck)
            return
        elif 'TrigMonomial' in globals() and type(s) == TrigMonomial:
            if len(s.symbols) != 0:
                raise ValueError("Unable to convert TrigMonomial to Constant")
            self.f = deepcopy(s.coe.f)
            self.cm = deepcopy(s.coe.cm)
            self.ck = deepcopy(s.coe.ck)
            return
        elif 'TrigPolynomial' in globals() and type(s) == TrigPolynomial:
            if len(s.terms) == 0:
                return
            elif len(s.terms) == 1:
                return self.__init__(s.terms[0])
            else:
                raise ValueError("Unable to convert TrigPolynomial to Constant")
        elif type(s) != str:
            raise TypeError("Unsupported Constant type: " + str(type(s)))

        try:
            self.f = PolyPi(s)
        except:
            raise NotImplementedError("Unsupported Constant string parsing")

    def __str__(self):
        terms = []
        if self.f != 0:
            terms.append(str(self.f))
        for i in range(len(self.cm)):
            s = str(self.cm[i]) + '*'
            if s == '1*':
                s = ''
            s = s + "cos(" + str(self.ck[i]) + ")"
            if s[0] != '-':
                s = '+' + s
            terms.append(s)
        if len(terms) == 0:
            return '0'
        return (''.join(terms)).lstrip('+')

    def simplify(self):
        n = min(len(self.cm), len(self.ck))
        m1 = self.cm[:n]
        k1 = self.ck[:n]
        for i in range(n):
            k1[i] = abs(k1[i]).normalize_0_2pi().simplify()
            if k1[i] in [PI_12, PI_32]:
                k1[i] = PolyPi(0)
                m1[i] = 0
            if k1[i] == PI_22:
                k1[i] = PolyPi(0)
                m1[i] = -1
        res = Constant(self.f.simplify())
        for i in range(n):
            if m1[i] == None:
                continue
            val = m1[i]
            for j in range(i+1, n):
                if k1[i] == k1[j]:
                    val += m1[j]
                    m1[j] = None
            if k1[i] == 0:
                res.f += val
            else:
                res.cm.append(deepcopy(val))
                res.ck.append(deepcopy(k1[i]))
        return res

    def __eq__(self, s):
        s = Constant(s).simplify()
        t = self.simplify()
        return t.f == s.f and t.cm == s.cm and t.ck == s.ck

    def __add__(a, b):
        c = Constant()
        if type(b) in [TrigMonomial, TrigPolynomial]:
            return b.__add__(a)
        b = Constant(b)
        c.f = a.f + b.f
        c.cm = a.cm + b.cm
        c.ck = a.ck + b.ck
        return c.simplify()

    def __neg__(a):
        c = Constant()
        c.f = -a.f
        c.cm = [-_ for _ in a.cm]
        c.ck = a.ck[:]
        return c

    def __sub__(a, b):
        return a.__add__(b.__neg__())

    def __mul__(a, b):
        if type(b) in [TrigMonomial, TrigPolynomial]:
            return b.__mul__(a)
        b = Constant(b)
        c = Constant()
        # a.f * b.f
        c.f = a.f * b.f
        # a.f * b.c
        for i in range(len(b.ck)):
            c.cm.append(a.f * b.cm[i])
            c.ck.append(b.ck[i])
        # b.f * a.c
        for i in range(len(a.ck)):
            c.cm.append(b.f * a.cm[i])
            c.ck.append(a.ck[i])
        # a.c * b.c
        for i in range(len(a.ck)):
            ka, ma = a.ck[i], a.cm[i]
            for j in range(len(b.ck)):
                kb, mb = b.ck[j], b.cm[j]
                # cos(x)*cos(y) = 1/2*cos(x+y) + 1/2*cos(x-y)
                c.cm.append(ma*mb * Fraction(1, 2))
                c.ck.append(ka + kb)
                c.cm.append(ma*mb * Fraction(1, 2))
                c.ck.append(ka - kb)
        return c.simplify()

    def __truediv__(a, b):
        b = Constant(b)
        c = Constant()
        if len(b.cm) != 0 or len(b.ck) != 0:
            raise ValueError("Unable to divide by trigonometric function")
        c.f = a.f / b.f
        c.cm = [m/b.f for m in a.cm]
        c.ck = deepcopy(a.ck)
        return c

    def __pow__(self, e: int):
        if e < 0:
            if self.cm == [] and self.ck == []:
                return pow(self.f, e)
            raise ValueError("Exponent must non-negative")
        r = Constant(1)
        x = deepcopy(self)
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
        return p.__mul__(q)

    def __rsub__(p, q):
        return Constant(q).__sub__(p)

    def __rtruediv__(p, q):
        return Constant(q).__truediv__(p)

    def is_integer(self):
        return len(self.ck) == 0 and self.f.is_integer()

    def __int__(self):
        return self.f.__int__()

    def __float__(self):
        """ Numerical evaluation """
        ans = float(self.f)
        for i in range(min(len(self.cm), len(self.ck))):
            m = float(self.cm)
            k = float(self.ck)
            ans += m*math.cos(k)
        return ans


PI = Constant('pi')


class TrigMonomial():

    """
        coe: Constant
        symbols: { 'x': [2, [3], [4, 5]] } means x^2*sin(3*x)*cos(4*x)*cos(5*x)
    """

    def __init__(self, s=1):
        self.coe = Constant(1)
        self.symbols = {}

        if type(s) in [int, float, Fraction, PolyPi, Constant]:
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
                            raise NotImplementedError(
                                "Multiple multiplication signs inside trigonometric")
                        u, v = t.split('*')
                        if u.lstrip('-').isnumeric() and v.isalnum():
                            k, x = int(u), v
                        elif u.isalnum() and v.lstrip('-').isnumeric():
                            k, x = int(v), u
                        else:
                            raise ValueError(
                                "Expect one integer and one variable name")
                        if not x in self.symbols:
                            self.symbols[x] = [0, [], []]
                        self.symbols[x][index] += [k]*p
                    else:  # cos(x), sin(-x)
                        k = -1 if t[0] == '-' else 1
                        t = t.lstrip('-')
                        if t.isnumeric():
                            raise NotImplementedError(
                                "Unsupported number inside trigonometric")
                        if not t.isalnum():
                            raise ValueError(f"Expect a variable name ({t})")
                        if not t in self.symbols:
                            self.symbols[t] = [0, [], []]
                        self.symbols[t][index] += [k]*p
                else:
                    raise ValueError(f"Unrecognized name {t}")

    def __str__(self):
        res = []
        sc = str(self.coe)
        if len(self.symbols) == 0:
            return sc
        if sc != '1':
            if sc == '-1':
                res.append('-')
            else:
                res.append(sc)
        for s in self.symbols:
            p = self.symbols[s][0]
            if p != 0:
                res.append(s if p == 1 else f"{s}**{self.symbols[s][0]}")
            for c in self.symbols[s][1]:
                sc = str(c) + '*'
                if sc == '1*':
                    sc = ''
                res.append(f"sin({sc}{s})")
            for c in self.symbols[s][2]:
                sc = str(c) + '*'
                if sc == '1*':
                    sc = ''
                res.append(f"cos({sc}{s})")
        return '*'.join(res).replace('-*', '-')

    def __mul__(a, b):
        c = TrigMonomial()
        b = TrigMonomial(b)
        c.coe = a.coe * b.coe
        for s in a.symbols:
            v = deepcopy(a.symbols[s])
            for t in b.symbols:
                if s == t:
                    w = b.symbols[t]
                    v[0] += w[0]
                    v[1] += w[1]
                    v[2] += w[2]
            c.symbols[s] = v
        for s in b.symbols:
            if s not in a.symbols:
                c.symbols[s] = deepcopy(b.symbols[s])
        return c

    def __truediv__(p, q):
        q = TrigMonomial(q)
        p.coe /= q.coe
        for s in q.symbols:
            if s not in p.symbols:
                raise ValueError("Unable to represent negative power")
            if len(q.symbols[s][1]) != 0 or len(q.symbols[s][2]) != 0:
                raise ValueError("Unable to divide by trigonometric expression")
            p.symbols[s][0] -= q.symbols[s][0]
            if p.symbols[s][0] < 0:
                raise ValueError("Unable to represent negative power")
        return p

    def simplify(self):
        res = TrigMonomial()
        if type(self.coe) != Constant:
            raise TypeError("TrigMonomial.coe is not a constant")
        res.coe = self.coe.simplify()
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

    def __rmul__(p, q):
        return p.__mul__(q)

    def subs(self, maps: dict):
        ans = TrigMonomial(self.coe)
        for s in self.symbols:
            if s in maps:
                if type(maps[s]) not in [int, float, Fraction, PolyPi, Constant]:
                    raise TypeError("Substitution type must be constant")
                p, ss, cs = self.symbols[s]
                c = maps[s]**p
                for k in ss:
                    c *= sin(k*maps[s])
                for k in cs:
                    c *= cos(k*maps[s])
                ans.coe *= Constant(c)
            else:
                ans.symbols[s] = self.symbols[s]
        if len(ans.symbols) == 0:
            return ans.coe
        else:
            return ans
    
    def __float__(self):
        if len(self.symbols) != 0:
            raise ValueError("Not a constant: TrigMonomial contains variables")
        ans = float(self.coe)
        return ans


class TrigPolynomial():

    def __init__(self, s=0):
        self.terms = []

        if type(s) == TrigPolynomial:
            self.terms = deepcopy(s.terms)
            return
        elif type(s) in [int, float, Fraction, PolyPi, Constant, TrigMonomial]:
            if s != 0:
                self.terms.append(TrigMonomial(s))
            return
        elif type(s) in [list, tuple]:
            self.terms = [TrigMonomial(t) for t in s]
        elif type(s) != str:
            raise TypeError("Unsupported TrigMonomial type: " + str(type(s)))

        # just lazy
        s = '(' + s + ')'
        s = s.replace('**', '^').replace('^-', '^_')
        s = s.replace('-', '+-').replace('(+', '(').replace('++', '+')
        s = s.replace('_', '-')
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
        p = q
        q = TrigPolynomial()
        i = 0
        while i < len(p.terms):
            s = p.terms[i]
            j = i+1
            while j < len(p.terms):
                if s.symbols == p.terms[j].symbols:
                    s.coe += p.terms[j].coe
                    p.terms = p.terms[0:j]+p.terms[j+1:len(p.terms)]
                else:
                    j += 1
            if s.coe != 0:
                q.terms.append(s)
            i += 1
        return q

    def __add__(p, q, simplify=True):
        if type(q) != TrigPolynomial:
            q = TrigPolynomial(q)
        r = TrigPolynomial()
        r.terms = p.terms + q.terms
        if simplify:
            r = r.simplify()
        return r

    def __neg__(p):
        q = deepcopy(p)
        for i in range(len(q.terms)):
            q.terms[i].coe *= -1
        return q

    def __sub__(p, q, simplify=True):
        if type(q) != TrigPolynomial:
            q = TrigPolynomial(q)
        return p.__add__(q.__neg__(), simplify)

    def __mul__(p, q, simplify=True):
        if type(q) != TrigPolynomial:
            q = TrigPolynomial(q)
        r = TrigPolynomial()
        for pi in p.terms:
            for qi in q.terms:
                r.terms.append(pi*qi)
        if simplify:
            r = r.simplify()
        return r

    def __pow__(p, e: int):
        if e < 0:
            raise ValueError("Exponent must be non-negative integer")
        r = TrigPolynomial(1)
        x = deepcopy(p)
        while e:
            if e & 1:
                r = r * x
            e >>= 1
            if e:
                x = x * x
        return r

    def __truediv__(p, q):
        p = deepcopy(p)
        if type(q) == TrigPolynomial:
            if len(q.terms) == 0:
                raise ZeroDivisionError(
                    "Unable to divide by empty TrigPolynomial")
            if len(q.terms) != 1:
                raise ValueError("Unable to divide by TrigPolynomial")
            q = q.terms[0]
        for i in range(len(p.terms)):
            p.terms[i] /= q
        return p

    def __radd__(p, q):
        return p.__add__(q)

    def __rmul__(p, q):
        return p.__mul__(q)

    def __rsub__(p, q):
        return TrigPolynomial(q).__sub__(p)

    def __rtruediv__(p, q):
        return TrigPolynomial(q).__truediv__(p)

    def subs(self, maps: dict):
        ans = TrigPolynomial()
        for t in self.terms:
            v = t.subs(maps)
            # ans += v
            ans.terms.append(TrigMonomial(v))
        return ans.simplify()
    
    def __float__(self):
        ans = 0.0
        for t in self.terms:
            v = float(t)
            ans += v
        return ans


def sin(x):
    if type(x) in [int, float, Fraction, PolyPi]:
        res = Constant()
        res.cm.append(1)
        res.ck.append(PI.f/2 - PolyPi(x))
        res = TrigPolynomial(res)
        return res.simplify()
    elif type(x) == Constant:
        if len(x.cm) != 0:
            raise ValueError("Unable to represent sin")
        return sin(x.f)
    elif type(x) == TrigMonomial:
        if len(x.symbols) == 0:
            return sin(x.coe)
        elif len(x.symbols) == 1:
            if not x.coe.is_integer():
                raise ValueError("Unable to represent sin")
            res = TrigMonomial()
            res.coe = Constant(1)
            for s in x.symbols:
                if x.symbols[s] != [1, [], []]:
                    raise ValueError("Unable to represent sin")
                res.symbols[s] = [0, [int(x.coe)], []]
            return TrigPolynomial(res)
        else:
            raise ValueError("Unable to represent sin")
    elif type(x) == TrigPolynomial:
        if len(x.terms) == 0:
            return 0
        elif len(x.terms) == 1:
            return sin(x.terms[0])
        else:
            raise NotImplementedError("Unsupported sin of sum of functions")


def cos(x):
    if type(x) in [int, float, Fraction, PolyPi]:
        res = Constant()
        res.cm.append(1)
        res.ck.append(PolyPi(x))
        return TrigPolynomial(res)
    elif type(x) == Constant:
        if len(x.cm) != 0:
            raise ValueError("Unable to represent cos")
        return cos(x.f)
    elif type(x) == TrigMonomial:
        if len(x.symbols) == 0:
            return cos(x.coe)
        elif len(x.symbols) == 1:
            if not x.coe.is_integer():
                raise ValueError("Unable to represent cos")
            res = TrigMonomial()
            res.coe = Constant(1)
            for s in x.symbols:
                if x.symbols[s] != [1, [], []]:
                    raise ValueError("Unable to represent cos")
                res.symbols[s] = [0, [], [int(x.coe)]]
            return TrigPolynomial(res)
        else:
            raise ValueError("Unable to represent cos")
    elif type(x) == TrigPolynomial:
        if len(x.terms) == 0:
            return 0
        elif len(x.terms) == 1:
            return cos(x.terms[0])
        else:
            raise NotImplementedError("Unsupported cos of sum of functions")


def fourier_series(c: list, s: list):
    """
        Not really.
        Rewrite: prod(cos(c*x))*prod(sin(s*x))
        To: sum(rc*cos(i*x))+sum(rs*sin(i*x))
        Return rc and rs.
        I believe this function has an O(N^2) time complexity.
    """
    for _ in c+s:
        if not (type(_) == int and _ > 0):
            raise ValueError("Harmonic must be positive integer")

    is_odd = len(s) % 2 == 1  # odd or even function
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


def integrate_polycos(p: int, k: int):
    """
        \int x^p \cos(kx) dx
        return two polynomial lists (pc, ps); pc(x)*cos(k*x)+ps(x)*sin(k*x)
    """
    if k == 0:
        pc = [Fraction(0)]*(p+1) + [Fraction(1, p+1)]
        return (pc, [0])
    if p == 0:
        return ([0], [Fraction(1, k)])

    # \int x^p\cos(kx)dx = \frac{1}{k} x^p\sin(kx) - \frac{p}{k} \int x^{p-1}\sin(kx)dx
    pc, ps = [Fraction(0)]*(p+1), [Fraction(0)]*(p+1)
    ps[p] += Fraction(1, k)
    rc, rs = integrate_polysin(p-1, k)
    for i in range(p):
        pc[i] -= Fraction(p, k) * rc[i]
        ps[i] -= Fraction(p, k) * rs[i]
    return (pc, ps)


def integrate_polysin(p: int, k: int):
    """
        \int x^p \sin(kx) dx
        return two polynomial lists (pc, ps); pc(x)*cos(k*x)+ps(x)*sin(k*x)
    """
    if k == 0:
        return ([0], [0])
    if p == 0:
        return ([Fraction(-1, k)], [0])

    # \int x^p\sin(kx)dx = -\frac{1}{k} x^p\cos(kx) + \frac{p}{k} \int x^{p-1}\cos(kx)dx
    pc, ps = [Fraction(0)]*(p+1), [Fraction(0)]*(p+1)
    pc[p] -= Fraction(1, k)
    rc, rs = integrate_polycos(p-1, k)
    for i in range(p):
        pc[i] += Fraction(p, k) * rc[i]
        ps[i] += Fraction(p, k) * rs[i]
    return (pc, ps)


def integrate_monomial(varname: str, symbol: list):
    """
        Receive something like ('x', [3, [1], [2]]) representing x**3*sin(x)*cos(2*x)
        Return a TrigPolynomial, its indefinite integral
    """
    p = symbol[0]
    ans = TrigPolynomial()
    if symbol[1] == [] and symbol[2] == []:
        a = TrigMonomial()
        a.coe = Constant(Fraction(1, p+1))
        a.symbols[varname] = [p+1, [], []]
        ans.terms.append(deepcopy(a))
        return ans

    fc, fs = fourier_series(symbol[2], symbol[1])
    if len(fc) > 0:
        f = fc
        integrate_polytrig = integrate_polycos
    if len(fs) > 0:
        f = fs
        integrate_polytrig = integrate_polysin
    for k in range(len(f)):
        if f[k] == 0:
            continue
        a = TrigMonomial()
        rc, rs = integrate_polytrig(p, k)
        for i in range(len(rc)):
            if rc[i] == 0:
                continue
            a.coe = Constant(rc[i]*f[k])
            a.symbols[varname] = [i, [], [k]]
            ans.terms.append(deepcopy(a))
        for i in range(len(rs)):
            if rs[i] == 0:
                continue
            a.coe = Constant(rs[i]*f[k])
            a.symbols[varname] = [i, [k], []]
            ans.terms.append(deepcopy(a))
    #return ans
    return ans.simplify()


def integrate(poly, args):
    if type(poly) == TrigMonomial:
        if type(args) != list:
            args = [args]
        if len(args) == 0:
            return poly
        a = args[0]
        ans = TrigPolynomial()
        if type(a) not in [str, tuple]:
            raise TypeError('Integral argument must be variable name or tuple')
        if type(a) == tuple:
            a, x0, x1 = a
            x0, x1 = Constant(x0), Constant(x1)
        else:
            x0 = x1 = None
        m = deepcopy(poly)
        if a in poly.symbols:
            m.symbols.pop(a)
            p = integrate_monomial(a, poly.symbols[a])
            ans = p * m
        else:
            m.symbols[a] = [1, [], []]
            ans.terms.append(m)
        if type(x0) == Constant and type(x1) == Constant:
            i0 = ans.subs({a: x0})
            i1 = ans.subs({a: x1})
            ans = i1 - i0
        return integrate(ans, args[1:])

    elif type(poly) == TrigPolynomial:
        p = deepcopy(poly).simplify()
        ans = TrigPolynomial()
        for t in p.terms:
            #ans += integrate(t, args)
            ans.terms += TrigPolynomial(integrate(t, args)).terms
        ans = ans.simplify()
        return ans
    else:
        return integrate(TrigPolynomial(poly), args)


# testing


def test1():
    x, y = TrigPolynomial('x'), TrigPolynomial('y')
    p = (x*sin(x)**2+x*y**2)**4
    print(p)
    i = integrate(p, [('x', 0, 2*PI), ('y', 0, 1)])
    print(i)
    print(float(i))


def test2():
    # speed test for noise function
    ss5 = lambda x: x*x*x*(10+x*(-15+x*6))
    ss5_grad = lambda x: ((x*30-60)*x+30)*x*x
    mix = lambda a, b, x: a+(b-a)*x

    NUMCOMP = 10
    varnames = [f's{i}' for i in range(NUMCOMP)]
    halton = [TrigPolynomial(s) for s in varnames]
    diffs = [(s, 0, 1) for s in varnames]

    z00 = 2*halton[0]-1
    z01 = 2*halton[1]-1
    z10 = 2*halton[2]-1
    z11 = 2*halton[3]-1
    x = ss5(halton[4])
    y = ss5(halton[5])
    v = mix(mix(z00, z01, x), mix(z10, z11, x), y)

    ans = integrate(v, diffs)
    print(ans)
    print(float(ans))


def test3():

    x, y = TrigPolynomial('x'), TrigPolynomial('y')
    r, a = TrigPolynomial('r'), TrigPolynomial('a')
    nx, ny = r*cos(a), r*sin(a)
    dot = nx*x+ny*y
    expr = dot*dot / r / (2*PI)
    var = integrate(expr, [('x', 0, 1), ('y', 0, 1), ('r', 0, 1), ('a', 0, 2*PI)])
    print(var)


if __name__ == "__main__":

    import sys
    #sys.stdout = open("D:/log.txt", 'w')

    import cProfile
    from time import perf_counter
    t0 = perf_counter()

    test3()
    #cProfile.run('test2()')

    t1 = perf_counter()
    sys.stderr.write("Time elapsed: {:.6f} secs".format(t1-t0))
    sys.stdout.close()

    
else:
    del test1, test2
