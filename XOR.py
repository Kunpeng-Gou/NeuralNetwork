import numpy as np
import NN

if __name__ == '__main__':
    train = [
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [1, 0, 0, 1]
    ]
    for i in range(4):
        train[0][i] = np.array(train[0][i])
    train[1] = np.array(train[1])
    cnt = 0
    for i in range(1):
        print(i)
        net = NN.NeuralNetwork(2, 2, 2, 5, 0.75, train, 4, 5000)
        net.back_propagation()
        ret = net.accuracy(train)
        print(ret)
        if ret == 1.0:
            cnt += 1
    print(cnt / 1)
    # print(r)

