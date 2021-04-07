import math


class Dual:

    def __init__(self, a, b=0):
        self.a = a
        self.b = b

    @classmethod
    def as_variable(cls, x):
        return cls(x, 1)

    def __add__(self, u):
        a = self.a + u.a
        b = self.b + u.b
        return Dual(a, b)

    def __sub__(self, u):
        a = self.a - u.a
        b = self.b - u.b
        return Dual(a, b)

    def __mul__(self, u):
        a = self.a * u.a
        b = self.a * u.b + self.b * u.a
        return Dual(a, b)

    def __repr__(self):
        return str(self.a) + " + " + str(self.b) + "ε"


def dual_ln(v):
    a = math.log(v.a)
    b = (1/v.a) * v.b
    return Dual(a, b)


def dual_cos(v):
    a = math.cos(v.a)
    b = -math.sin(v.a) * v.b
    return Dual(a, b)


# Let us compute log(5 * x + 5) + 3(x + 2)**2
def h(x):
    x = Dual(x, 1)
    v1 = Dual(5) * x + Dual(5)
    v2 = dual_ln(v1)
    v3 = x + Dual(2)
    v3 = Dual(3) * v3 * v3
    return v2 + v3

def g(x):
    v0 = Dual(x, 1)
    v1 = v0 * v0
    v2 = Dual(5) * v1
    v3 = dual_cos(v2)
    return v3


def f(x, y):
    v1 = dual_cos(x)
    v2 = dual_ln(x)
    v3 = x * y
    v4 = y * y * y
    v5 = v1 + v3
    v6 = v4 + v2
    v7 = v6 - y
    return v5, v7
