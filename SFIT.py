from os import times
import numpy as np
import cv2
import math
import logging


class SFIT():
    def __init__(self, image, args) -> None:
        '''
        SFIT算法
        复现原文算法

        高斯金字塔每一层:Layer
        每一层中的图像：image
        高斯金字塔从下到上的每一层图像列表：image_list
        高斯金字塔从下到上的每一层高斯核列表，kernel_list
        '''
        self.image = 1
        self.args = 1
        self.sigma = 1
        self.threshold_const =1 
        self.border =1
    
    def get_basic_image(self, img, targe_blur, scr_blur=0.5):
        '''
        假如希望给图像一个blur
        需要减去原始图片自身0.5的模糊
        还原计算初始base图像，用于每个层的第一层
        '''
        upsample_img = cv2.resize(img, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        blur_kernel = cv2.sqrt(max((targe_blur**2) - (scr_blur**2), 0.0001))
        blur_img = cv2.GaussianBlur(img, (0,0), sigmaX=blur_kernel, sigmaY=blur_kernel)
        return blur_img
    
    def get_pyramid_layer_num(self, img_shape):
        '''
        计算金字塔的层数
        '''
        min_size = min(img_shape)
        O = round(cv2.log(min_size) / cv2.log(2))
        O = O-1
        return int(O)

    def get_blur_kernels_list(self,image_num):
        '''
        获取每层每个图像对应的blur核
        '''
        k = 2**(1./ image_num)
        image_num = image_num + 3
        blur_kernels_list = np.zeros(image_num)
        blur_kernels_list[0] = self.sigma

        for index in range(1, image_num):
            pre = k**(index-1)* self.sigma
            now = k * pre
            blur_kernels_list[index] = now

        return blur_kernels_list
    
    def get_blur_image_list(self, image, layer_num, blur_kernel_list):
        '''
        对高斯金字塔图像进行整理
        '''
        blur_image_list = []
        for layer_index in range(layer_num):
            images_each_layer = [image]
            for blur_kernel in blur_kernel_list[1:]:
                new_image = cv2.GaussianBlur(image, (0,0), sigmaX= blur_kernel, sigmaY=blur_kernel)
                images_each_layer.append(new_image)
            blur_image_list.append(images_each_layer)
           
            #准备下一个层中的第一个图像,选取这一层图像的倒数第三个图像直接下采样
            image_2_next_layer = images_each_layer[-3]
            new_w =int(image_2_next_layer.shape[0]/2)
            new_h = int(image_2_next_layer.shape[1]/2)
            image = cv2.resize(image_2_next_layer, new_w, new_h, interpolation=cv2.INTER_LINEAR)

        blur_image_list = np.array(blur_image_list)
        return blur_image_list

    def get_DoG_image_list(self, image_list):
        '''
        高斯金字塔中相邻图像相减获得高斯差分图像
        '''
        DoG_image_list = []
        
        for images_each_layer in image_list:
            DoG_diff_list = []
            for index in range(1,len(image_list)):
                pre_image = image_list[index-1]
                this_image = image_list[index]
                DoG_diff_list.append(this_image-pre_image)
            DoG_image_list.append(DoG_diff_list)
        
        DoG_image_list = np.array(DoG_image_list)
        return DoG_image_list
    
    def get_keypoints(self, blur_image_list, DoG_image_list, layer_num):
        '''
        在金字塔中的找到极值点。
        上一层和下一层相邻的点分别都有9个
        自身所在层的相邻点有8个（除自己之外）

        blur_image_list:高斯图像从下往上
        DoG_image_list:金字塔差分图像从下往上
        layer_num:金字塔堆叠的层数
        sigma:高斯核参数
        border:卷积边界
        threshold:选择极值点的阈值
        '''
        threshold = np.floor( 0.5 * self.threshold_const / layer_num * 255)
        keypoints = []

        for layer_index, DoG_each_layer in enumerate(DoG_image_list):
            for image_index in range(3,len(DoG_each_layer)):
                bottom_image = DoG_each_layer[image_index-2]
                mid_image = DoG_each_layer[image_index-1]
                top_image = DoG_each_layer[image_index]
                image = np.array([bottom_image,mid_image,top_image])
        


    def conv3X3(self, image, stride=3):
        '''
        对图像进行卷积
        '''
        assert (stride%2 ==0)

        W,H = image.shape[1],image.shape[2]
        for y in range(self.border, H-self.border):
            for x in range(self.border, W-self.border):
                if self.is_keypoint(image[:,y-1:y-1+stride, x-1:x-1+stride]):
                    #离散的点不能求最优值，用拟合函数来计算
                    real_result = self.get_real_result()


    def is_keypoint(self, image):
        '''
        判断对应坐标的图像是否是关键极值点
        '''
        return 1
    
    def get_real_result(self, 
                        x, y, 
                        image_index, 
                        layer_index, 
                        image_num_per_layer,
                        cube,
                        ratio = 10,
                        try_times= 5):
        '''
        用函数对极值点进行判断
        离散的点无法确定极值点
        需要对主要的点判断偏移量

        x,y 是坐标
        '''

        canFind = False

        shape = cube[0].shape
        for t in range(try_times):
            pixel_cube = cube.astype('float32') / 255
            gradient = self.get_gradient_from_cube(pixel_cube)
            hessian  = self.get_hessian_from_cube(pixel_cube)
            #进行最小二乘线性拟合，找到合适点
            shift = - np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            #最优点的偏移不会偏移到其他像素点就自动结束
            if np.abs(shift[0])<0.5 and np.abs(shift[1])<0.5 and np.abs(shift[2])<0.5:
                break
            
            #这部分可能有bug
            y += int(round(shift[0]))
            x += int(round(shift[1]))
            image_index = int(round(shift[2]))

            #查看是否极值点能落在原始图片区域里
            if y < self.border and y >= shape[0]-self.border and x<self.border and x>shape[1]-self.border:
                canFind = True
                break
            
            #找不到最优解
            if canFind:
                return None
            
            #优化到最低
            if t >= try_times-1:
                return None
            
            #解
            keypoint = pixel_cube[1,1,1]+ 0.5*gradient*shift

            if np.abs(keypoint) * image_num_per_layer >= self.threshold_const:
                xy_h = hessian[:2, :2]
                #求对角线元素之和
                xy_h_trace = np.trace(xy_h)
                xy_h_det = np.linalg.det(xy_h)
                # 对比检查
                if 






        
    

    def run(self, run):
        return 1


if __name__ == '__main__':
    logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %p',
    level=logging.INFO)

    logging.info('开始！')
    logging.info(f'args:')
