import csv
import math
import numpy as np
import matplotlib.pyplot as plt


def read_csv(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Month']))
            y.append(float(row['Temp']))
    return x, y


def tabulate(x, y, filename):
    with open(filename, 'w') as file:
        for i in range(len(x)):
            file.write("%.10f %.10f\n" % (x[i], y[i]))


def read_table(filename):
    x = []
    y = []
    with open(filename, 'r') as file:
        for line in file:
            a, b = line.split()
            x.append(float(a))
            y.append(float(b))
    return x, y


def form_B(x, m):
    B = [[0.0] * (m + 1) for _ in range(m + 1)]
    for k in range(m + 1):
        for l in range(m + 1):
            s = 0.0
            for i in range(len(x)):
                s += x[i] ** (k + l)
            B[k][l] = s
    return B


def form_C(x, y, m):
    C = [0.0] * (m + 1)
    for k in range(m + 1):
        s = 0.0
        for i in range(len(x)):
            s += y[i] * x[i] ** k
        C[k] = s
    return C


def gauss_solve(B, C):
    n = len(C)
    A = [row[:] for row in B]
    f = C[:]
    for k in range(n):
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[max_row][k]):
                max_row = i
        A[k], A[max_row] = A[max_row], A[k]
        f[k], f[max_row] = f[max_row], f[k]
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            f[i] -= factor * f[k]
    a = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += A[i][j] * a[j]
        a[i] = (f[i] - s) / A[i][i]
    return a


def polynomial(t, a):
    res = 0.0
    for j in range(len(a)):
        res += a[j] * t ** j
    return res


def variance(x, y, a):
    s = 0.0
    for i in range(len(x)):
        s += (polynomial(x[i], a) - y[i]) ** 2
    return math.sqrt(s / len(x))


def error(t, f, a):
    return abs(f - polynomial(t, a))


tabulate(*read_csv("temp.csv"), "temp.txt")
x, y = read_table("temp.txt")
n = len(x) - 1

degrees = list(range(1, 11))
variances = []
coefs = []
for m in degrees:
    a = gauss_solve(form_B(x, m), form_C(x, y, m))
    coefs.append(a)
    variances.append(variance(x, y, a))
    print("m = %2d, delta = %.6f" % (m, variances[-1]))

m_opt = degrees[variances.index(min(variances))]
a_opt = coefs[m_opt - 1]
print("\nОптимальний степінь m = %d" % m_opt)
print("Коефіцієнти:", a_opt)

x_future = [25, 26, 27]
for t in x_future:
    print("Прогноз: місяць %d, температура %.4f" % (t, polynomial(t, a_opt)))

plt.figure(figsize=(8, 5))
plt.plot(degrees, variances, 'o-')
plt.xlabel('m')
plt.ylabel('delta')
plt.title("Залежність дисперсії від степеня многочлена")
plt.grid()
plt.savefig("variance.png")
plt.show()

h1 = (x[-1] - x[0]) / (20 * n)
tt = [x[0] + i * h1 for i in range(20 * n + 1)]

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'ro', label='дані')
plt.plot(tt, [polynomial(t, a_opt) for t in tt], label='m = %d' % m_opt)
plt.plot(x_future, [polynomial(t, a_opt) for t in x_future], 'g*', markersize=12, label='прогноз')
plt.xlabel('Місяць')
plt.ylabel('Температура')
plt.legend()
plt.grid()
plt.savefig("approx.png")
plt.show()

plt.figure(figsize=(8, 5))
for m in degrees:
    a = coefs[m - 1]
    eps = [error(t, float(np.interp(t, x, y)), a) for t in tt]
    plt.plot(tt, eps, label='m = %d' % m)
plt.xlabel('Місяць')
plt.ylabel('eps')
plt.title("Похибка апроксимації")
plt.legend()
plt.grid()
plt.savefig("error.png")
plt.show()