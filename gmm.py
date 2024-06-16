import argparse
import os
import cv2
import numpy as np


class GMM():
    def __init__(self, img, args):
        height, width, channel = img.shape
        self.GS_NUM = args.GS_NUM  # 高斯分布个数
        # 初始化
        self.mean = np.zeros(shape=(args.GS_NUM, height, width, channel))
        self.deviation = np.ones(shape=(args.GS_NUM, height, width))
        self.weight = np.ones(shape=(args.GS_NUM, height, width)) / args.GS_NUM

    def train(self, img, args):
        height, width, channel = img.shape
        self.threshold = args.threshold  # 阈值
        self.alpha = args.alpha  # 更新率

        # 判断在哪几个高斯分布内
        score = -1 * (self.weight / self.deviation)
        index = score.argsort(axis=0)
        score.sort(axis=0)
        score = -score
        self.bg = (score < self.threshold)
        self.bg = np.choose(index, self.bg)
        #  更新高斯分布权重
        gs_update = np.where(self.bg == 1, self.bg, -1)  # 不符合的高斯分布需要进行替换
        self.weight = np.where(gs_update == 1, (1 - self.alpha) * self.weight + self.alpha, self.weight)
        self.weight = np.where(gs_update == -1, self.alpha, self.weight)

        # 像素值是否在某一个高斯分布内
        diff = abs(img - self.mean)
        diff = np.mean(diff ** 2, axis=3) / (self.deviation ** 2)
        self.prob = np.exp(diff / (-2)) / (np.sqrt((2 * np.pi) ** 3) * self.deviation)  # 概率密度计算
        self.match_point = (np.sqrt(diff) < 2.5 * self.deviation)  # 计算在高斯分布内的像素点

        # 更新高斯分布
        new_mean = np.stack([img] * self.GS_NUM, axis=0)
        new_gs = np.stack([gs_update] * channel, axis=3)
        r = np.stack([self.prob] * channel, axis=3)
        # 更新均值
        self.mean = np.where(new_gs == 1, (1 - r) * self.mean + r * new_mean, self.mean)
        self.mean = np.where(new_gs == -1, new_mean, self.mean)
        # 更新标准差
        self.deviation = np.where(gs_update == 1,
                                  np.sqrt((1 - self.prob) * (self.deviation ** 2) + self.prob * (
                                      np.mean(np.subtract(img, self.mean) ** 2, axis=3))),
                                  self.deviation)
        self.deviation = np.where(gs_update == -1, 3 + np.ones(shape=(self.GS_NUM, height, width)), self.deviation)

    def test(self, img):
        height, width, channel = img.shape
        background = np.zeros(shape=(height, width), dtype=np.uint8)  # 背景用黑色
        foreground = 255 + np.zeros(shape=(height, width), dtype=np.uint8)  # 前景用白色
        # 判断一个点是否是背景点
        b = np.bitwise_and(self.bg, self.match_point)
        b = np.bitwise_or.reduce(b, axis=0)
        # 结果展示
        res = np.where(b == True, background, foreground)
        res = cv2.erode(res, kernel=(3, 3), iterations=1)
        cv2.imshow('res', res)
        return res

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--GS_NUM', type=int, default=5)
parser.add_argument('--threshold', type=float, default=0.7)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--iuput', type=str, default='input')  #TODO
parser.add_argument('--output', type=str, default=None)  # TODO


args = parser.parse_args()

def main():
    dir = args.input
    filelist = os.listdir(args.input)
    for file in filelist:
        filename = os.path.join(dir, file)
        img = cv2.imread(filename)
        gmm = GMM(img)
        break

    for file in filelist:
        filename = os.path.join(dir, file)
        img = cv2.imread(filename)
        gmm.train(img)
        res = gmm.test(img)
        if args.output is not None:
            output = os.path.join(args.output, file)
            cv2.imwrite(output, res)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()