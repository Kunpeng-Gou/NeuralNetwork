import numpy as np


if __name__ == '__main__':
    n = 2 * np.random.rand(3, 4) - 1
    v = np.random.rand(3)
    v = np.concatenate((v, [1]))
    # print(n, v)
    # r = np.matmul(n, v.T)
    r = np.array([1, 2, 3])
    print(r)
    a = 1 / (1 + np.exp(-r))
    print(a)
    print(np.zeros(10))
    l = list(range(10))
    print(list(reversed(l)))

    print()

    a = np.array(range(3))
    b = np.array(range(-4, 0))
    print(a, b)
    print(np.outer(a, b))
    print(np.outer(b, a))
    print(a.T.shape, a.shape)
    print(a[:2])

    print(np.linalg.norm(a))

    print(np.random.randn(4, 4))


