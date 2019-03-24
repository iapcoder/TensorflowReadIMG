# -*- coding: utf-8 -*-

"""
--------------------------------------------------------
# @Version : python3.7
# @Author  : wangTongGen
# @File    : readIMG.py
# @Software: PyCharm
# @Time    : 2019/3/24 09:22
--------------------------------------------------------
# @Description: 
--------------------------------------------------------
"""

import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOGO_LEVEL"] = "2"

def readIMG(file_list):
    """
    :param file_list: 文件路径列表
    :return: 每张图片的张量
    """
    # 1.构造文件队列
    file_queue = tf.train.string_input_producer(file_list, shuffle=False) # shuffle:True表示随机打乱顺序

    # 2.构造图像阅读器去读取图片内容(默认读取一张图片)
    reader = tf.WholeFileReader()

    key, value = reader.read(file_queue) # key:文件名 value:文件内容

    # 3.对读取的数据进行解码
    image = tf.image.decode_jpeg(value)

    # 4.处理图片的大小(使得每个样本的height,width一样)
    image_resize = tf.image.resize_images(image, [100,100])

    # 5.固定样本形状[200,200,3],批处理文件时要求明确数据形状,彩色图片为3，灰度图像为1
    image_resize.set_shape([100,100,3])

    # 6.批处理文件
    image_batch = tf.train.batch([image_resize], batch_size=4, num_threads=1, capacity=4)

    return image_batch


if __name__ == '__main__':

    file_name_list = os.listdir("../datas/animal")
    file_list = [os.path.join("../datas/animal", i) for i in file_name_list]
    image_batch = readIMG(file_list)

    with tf.Session() as sess:

        # 定义一个线程协调器
        coord = tf.train.Coordinator()
        # 开启读文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        img_data = sess.run(image_batch)
        print(img_data)

        coord.request_stop()
        coord.join(threads)