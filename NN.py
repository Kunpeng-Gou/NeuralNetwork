from time import time
import numpy as np
import read_mnist as rm


class Layer(object):
    def __init__(self, layer_name, input_num=1, output_num=1):
        self.layer_name = layer_name
        self.weights = np.random.randn(output_num, input_num + 1)
        # self.weights = (2 * np.random.rand(output_num, input_num + 1) - 1)
        # self.weights = 2 * np.sqrt(6 / (input_num + output_num + 1)) * (np.random.rand(output_num, input_num + 1) - 1)
        for row in self.weights:
            row[input_num] = 0
        self.input = None
        self.z = None
        self.x = None

        # self.print_weights()

    def feedforward(self, test):
        self.input = test
        if self.layer_name == 'input':
            self.z = test
            self.x = Layer.sigmoid(test)
            self.x = np.concatenate((self.x, [1]))
        elif self.layer_name == 'output':
            self.z = np.matmul(self.weights, self.input.T)
            self.x = Layer.sigmoid(self.z)
        else:
            self.z = np.matmul(self.weights, self.input.T)
            self.x = Layer.sigmoid(self.z)
            self.x = np.concatenate((self.x, [1]))
        return self.x

    def set_weights(self, weights):
        pass

    def print_weights(self):
        print(self.weights)

    @staticmethod
    def sigmoid(ret):
        return 1 / (1 + np.exp(-ret))

    @staticmethod
    def relu(ret):
        return max(0, ret)


class NeuralNetwork(object):
    def __init__(self, hidden_num, input_num, output_num, hidden_neuron_num, learning_ratio, train_data, batch_size=100,
                 epoch=50):
        self.train_data = train_data
        self.batch_size = batch_size
        layers = [Layer('input'), Layer('hidden', input_num, hidden_neuron_num)]
        for i in range(hidden_num - 1):
            layers.append(Layer('hidden', hidden_neuron_num, hidden_neuron_num))
        layers.append(Layer('output', hidden_neuron_num, output_num))
        self.layers = layers
        self.output_num = output_num
        self.epoch = epoch
        self.learning_ratio = learning_ratio

    def forward_propagation(self, test):
        ret = test
        for layer in self.layers:
            ret = layer.feedforward(ret)
        # print(ret)
        return ret

    def predict(self, test):
        ret = self.forward_propagation(test)
        return np.argsort(-ret)[0]

    def back_propagation(self):
        # self.batch_size = 60000
        # train_batch = self.train_data

        print(self.loss_function(self.train_data))
        for n in range(self.epoch):
            # print(n)
            for k in range(int(len(self.train_data[0]) / self.batch_size)):
                # print(k)
                train_batch = (self.train_data[0][k * self.batch_size:(k + 1) * self.batch_size], self.train_data[1][k * self.batch_size:(k + 1) * self.batch_size])
                batch_gradient = list(reversed([np.zeros(layer.weights.shape) for layer in self.layers]))
                layers = list(reversed(self.layers))

                print(self.loss_function(train_batch))
                for i in range(self.batch_size):
                    # print(i)
                    ret = self.forward_propagation(train_batch[0][i])
                    y = np.zeros(self.output_num)
                    y[train_batch[1][i]] = 1
                    # print(ret, y)
                    grad = []
                    loss_to_z = None

                    for j in range(len(layers)):
                        if layers[j].layer_name == 'input':
                            grad.append(np.array([1]))
                        elif layers[j].layer_name == 'output':
                            loss_to_z = a = (ret - y) * ret * (1 - ret)
                            b = layers[j].input
                            # print(loss_to_z, b)
                            gra = np.outer(a, b)
                            grad.append(gra)
                        else:
                            num = len(layers[j].x)
                            loss_to_z = np.matmul(layers[j - 1].weights.T, loss_to_z)[:num - 1]
                            a = loss_to_z * (layers[j].x * (1 - layers[j].x))[:num - 1]
                            b = layers[j].input
                            gra = np.outer(a, b)
                            grad.append(gra)

                    for j in range(len(batch_gradient)):
                        batch_gradient[j] += grad[j] / self.batch_size
                for i in range(len(layers)):
                    layers[i].weights -= self.learning_ratio * batch_gradient[i]
                self.layers = list(reversed(layers))

                # print(batch_gradient)
                print(self.loss_function(train_batch))
                # print(self.accuracy(self.train_data))
                print()

            # self.print_weights()
            print(self.loss_function(self.train_data))
            # print(self.accuracy(self.train_data))

    def accuracy(self, data):
        cnt = 0
        for i in range(len(data[0])):
            if self.predict(data[0][i]) == data[1][i]:
                cnt += 1
        return cnt / len(data[0])

    def loss_function(self, data):
        loss = 0
        for i in range(len(data[0])):
            ret = self.forward_propagation(data[0][i])
            y = np.zeros(self.output_num)
            y[data[1][i]] = 1
            loss += np.linalg.norm(ret - y)
        return loss

    def print_weights(self):
        for layer in self.layers:
            if layer.layer_name == 'input':
                continue
            layer.print_weights()


if __name__ == '__main__':
    start = time()
    train = rm.read_mnist('train')
    train = rm.process(train)
    my_network = NeuralNetwork(1, 784, 10, 200, 0.5, train, batch_size=100, epoch=50)
    test = rm.read_mnist('test')
    test = rm.process(test)

    count = 0
    for i in range(len(train[0])):
        # print(my_network.predict(train[0][i]), train[1][i])
        if my_network.predict(train[0][i]) == train[1][i]:
            count += 1
    print(count / len(train[0]))

    count = 0
    for i in range(len(test[0])):
        if my_network.predict(test[0][i]) == test[1][i]:
            count += 1
    print(count / len(test[0]))

    my_network.back_propagation()

    count = 0
    for i in range(len(train[0])):
        # print(my_network.predict(train[0][i]), train[1][i])
        if my_network.predict(train[0][i]) == train[1][i]:
            count += 1
    print(count / len(train[0]))

    count = 0
    for i in range(len(test[0])):
        if my_network.predict(test[0][i]) == test[1][i]:
            count += 1
    print(count / len(test[0]))
    print('finished in %fs' % (time() - start))
