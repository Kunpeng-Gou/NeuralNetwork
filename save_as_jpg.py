import cv2
import read_mnist as rm


if __name__ == '__main__':
    images = rm.read_mnist('train')

    # 将train_images转化为jpg格式的图片
    dir = 'mnist_as_jpg/train/'
    for i in range(len(images[0])):
        name = dir + '%d_%05d.jpg' % (images[1][i], i)
        cv2.imwrite(name, images[0][i].reshape(28, 28))
        print(i, name)

    print('?')
