# A file created to be used instead of sympy.integrate
# Expected to be faster than SymPy

from fractions import Fraction
from copy import deepcopy


class Monomial():

    def __init__(self, s=''):
        self.coe = Fraction(1)  # leading coefficient
        self.symbols = {}  # symbols and powers
        if type(s) in [int, float, Fraction]:
            self.coe = Fraction(s)
            return
        elif type(s) == Monomial:
            self.coe = s.coe
            self.symbols = deepcopy(s.symbols)
            return
        elif type(s) != str:
            raise TypeError("s needs to be a string")

        s = s.replace(' ', '')
        if s == '':
            return
        for t in s.replace('**', '^').split('*'):
            if '^' in t:
                t, p = t.split('^')
                p = int(p)
            else:
                p = 1
            if not type(p) == int:
                raise ValueError("Power must be an integer")
            if t.lstrip('-').replace('/', '').isnumeric():
                self.coe *= Fraction(t)**p
            elif t.isalnum() and (not t[0].isnumeric()):
                if t in self.symbols:
                    self.symbols[t] += p
                else:
                    self.symbols[t] = p
            else:
                raise ValueError(f"Invalid symbol {t}")

    def __str__(self):
        ans = [str(self.coe)]
        for s in self.symbols:
            p = self.symbols[s]
            if p == 1:
                ans.append(s)
            else:
                ans.append(f"{s}**{p}")
        return '*'.join(ans)

    def __mul__(a, b):
        c = Monomial()
        c.coe = a.coe * b.coe
        for s in a.symbols:
            v = a.symbols[s]
            for t in b.symbols:
                if s == t:
                    v += b.symbols[t]
            if v != 0:
                c.symbols[s] = v
        for s in b.symbols:
            if s not in a.symbols:
                c.symbols[s] = b.symbols[s]
        return c

    def subs(self, maps: dict):
        ans = Monomial()
        ans.coe *= self.coe
        for s in self.symbols:
            if s in maps:
                ans.coe *= maps[s]**self.symbols[s]
            else:
                ans.symbols[s] = self.symbols[s]
        if len(ans.symbols) == 0:
            return ans.coe
        else:
            return ans


class Polynomial():

    def __init__(self, s=''):
        if type(s) != str:
            self.terms = [Monomial(s)]
            return
        if s == '':
            self.terms = []
        else:
            s = s.replace('-', '+-').replace('++', '+')
            s = s.replace('^+-', '^-').replace('*+-', '*-')
            s = s.lstrip('+').split('+')
            self.terms = [Monomial(si) for si in s]

    def __str__(self):
        if len(self.terms) == 0:
            return '0'
        s = '+'.join([str(si) for si in self.terms])
        return s.replace('+-', '-')

    def simplify(self):
        p = deepcopy(self)
        q = Polynomial()
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

    def __add__(p, q):
        if type(q) != Polynomial:
            q = Polynomial(q)
        r = Polynomial()
        r.terms = p.terms + q.terms
        return r.simplify()

    def __neg__(p):
        q = deepcopy(p)
        for i in range(len(q.terms)):
            q.terms[i].coe *= -1
        return q

    def __sub__(p, q):
        if type(q) != Polynomial:
            q = Polynomial(q)
        return p.__add__(q.__neg__())

    def __mul__(p, q):
        if type(q) != Polynomial:
            q = Polynomial(q)
        r = Polynomial()
        for pi in p.terms:
            for qi in q.terms:
                r.terms.append(pi*qi)
        return r.simplify()

    def __pow__(p, e: int):
        r = Polynomial('1')
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
        return Polynomial(q).__mul__(p)

    def subs(self, maps: dict):
        ans = Polynomial()
        for t in self.terms:
            ans += Polynomial(t.subs(maps))
        return ans


def integrate(poly, args):
    poly = deepcopy(poly)
    if type(poly) == Monomial:
        for a in args:
            if type(a) == str:
                if a in poly.symbols:
                    if poly.symbols[a] == -1:
                        raise ZeroDivisionError("Integral of -1th power")
                    poly.symbols[a] += 1
                    poly.coe /= poly.symbols[a]
                else:
                    poly.symbols[a] = 1
            elif type(a) == tuple:
                a, x0, x1 = a
                if a in poly.symbols:
                    if poly.symbols[a] == -1:
                        raise ZeroDivisionError("Integral of -1th power")
                    deg = poly.symbols[a] + 1
                    poly.coe *= Fraction((x1**deg-x0**deg), deg)
                    poly.symbols.pop(a)
                else:
                    poly.coe *= x1-x0
            else:
                raise TypeError('Integral bound must be string or tuple')
        return poly
    elif type(poly) == Polynomial:
        for i in range(len(poly.terms)):
            poly.terms[i] = integrate(poly.terms[i], args)
        poly = poly.simplify()
        return poly
    else:
        return integrate(Polynomial(poly), args)


# testing

if __name__ == "__main__":

    p = Polynomial("x+y")*1
    q = 4+Polynomial("x*y")
    ans = integrate((2*p-q)**2, [('x', 0, 2), ('y', -1, 1), ('z', -1, 1)])
    print(ans)
