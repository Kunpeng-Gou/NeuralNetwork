# 这个模块实现对mnist文件的读取、导入和处理操作
import numpy as np

# 文件名
TRAIN_I = 'mnist/train-images.idx3-ubyte'
TRAIN_L = 'mnist/train-labels.idx1-ubyte'
TEST_I = 'mnist/t10k-images.idx3-ubyte'
TEST_L = 'mnist/t10k-labels.idx1-ubyte'
# 文件数据开始的位置
START_I = 16
START_L = 8
# 每个有意义数据单元的大小
SIZE_I = 784
SIZE_L = 1


def read_mnist(set_name):
    if set_name == 'train':
        images = my_read(TRAIN_I)
        labels = my_read(TRAIN_L)
    elif set_name == 'test':
        images = my_read(TEST_I)
        labels = my_read(TEST_L)
    else:
        return [], []
    return images, labels


def my_read(file_name):
    with open(file_name, 'rb') as file:
        data = file.read()

    magic_number = int.from_bytes(data[:4], byteorder='big')
    num = int.from_bytes(data[4:8], byteorder='big')

    if magic_number == 2049:
        start = START_L
        size = SIZE_L
    elif magic_number == 2051:
        start = START_I
        size = SIZE_I
    else:
        return None

    data_np = np.zeros((num, size), dtype='uint16')
    for i in range(num):
        arr = [byte for byte in data[start + i * size:start + i * size + size]]
        data_np[i] = arr
    if magic_number == 2049:
        data_np = data_np.reshape(num)
        # data_np = (data_np.reshape(1, num))[0]
    # print(data_np)
    return data_np


def process(data):
    return data[0] / 255, data[1]


if __name__ == '__main__':
    t = read_mnist('train')
    process(t)
    print(t)

