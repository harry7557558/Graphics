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
            raise TypeError("Unsupported Monomial type: " + str(type(s)))

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
        


# testing

if __name__ == "__main__":

    print(TrigMonomial("3*x*cos(x)**2*sin(-99*y)*cos(-x)*sin(-y)**2*cos(0*x)**3").simplify())
