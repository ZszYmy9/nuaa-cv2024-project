import cv2
import numpy as np
import argparse

class Haze_remove():
    def __init__(self, args):
        self.patch = args.patch
        self.w = args.w
        self.t0 = args.t0

    def read_file(self, img_root):
        self.img = cv2.imread(img_root) / 255
        self.img_gray = cv2.imread(img_root, 0) / 255

    def dark_channel(self, img):
        '''
        Get Dark Channel
        '''
        r, g, b = cv2.split(img)
        img_min = cv2.min(r, cv2.min(g, b))  # 暗通道
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.patch, self.patch))
        dark_img = cv2.erode(img_min, kernel)  # 块计算
        return dark_img

    def cal_atmo(self, mean=True):
        '''
        Estimate atmosphere light A
        '''
        flat = self.dark_channel(self.img).flatten()
        num = flat.shape[0] // 1000  # 最亮的0.1%像素
        indexs = flat.argsort()[-num:]
        cols = self.dark_channel(self.img).shape[1]
        pos = [(index // cols, index % cols) for index in indexs]  # 得到亮像素对应的索引
        if mean:
            points = np.array([self.img[p] for p in pos])
            atmo_light = points.mean(axis=0)  # 计算平均值
            return atmo_light
        pos = sorted(pos, key=lambda p: sum(self.img[p]), reverse=True)
        p = pos[0]
        self.atmo_light = self.img[p]
        return self.atmo_light

    def cal_t(self):
        '''
        Calculate transmittance
        '''
        tm = self.img / self.cal_atmo()  # 透射率
        self.tm = 1 - self.w * self.dark_channel(tm)  # 保留一点雾
        return self.tm

    def dehaze(self, output=None):
        '''
        dehaze
        '''
        self.result = np.empty_like(self.img)
        self.tm = cv2.max(self.tm, self.t0)  # 设置阈值
        for i in range(3):
            self.result[:, :, i] = (self.img[:, :, i] - self.cal_atmo()[i]) / self.tm[i] + self.cal_atmo()[i]  # 去雾
        cv2.imshow("source", self.img)
        cv2.imshow("result", self.result)
        cv2.waitKey()  # 退出观察
        if output is not None:
            cv2.imwrite(output, self.result * 255)  # 是否保存结果图片

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--patch', type=int, default=150)
parser.add_argument('--w', type=float, default=0.95)
parser.add_argument('--t0', type=float, default=0.35)
parser.add_argument('--img_root', type=str, default='haze_2.png')  #TODO
parser.add_argument('--output', type=str, default=None)  # TODO

args = parser.parse_args()

if __name__ == '__main__':
    Dehaze = Haze_remove(args=args)
    Dehaze.read_file(args.img_root)
    Dehaze.cal_atmo()
    Dehaze.cal_t()
    Dehaze.dehaze(args.output)

