import numpy as np
import cv2
import math
import logging
from functools import cmp_to_key

image_num_per_layer=3
args = 1
sigma = 1.6
threshold_const = 0.04
border =5

def get_basic_image(img, targe_blur, scr_blur=0.5):
        '''
        假如希望给图像一个blur
        需要减去原始图片自身0.5的模糊
        还原计算初始base图像，用于每个层的第一层
        '''
        upsample_img = cv2.resize(img, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        blur_kernel = np.sqrt(max((targe_blur**2) - (scr_blur**2 * 4), 0.0001))
        blur_img = cv2.GaussianBlur(upsample_img, (0,0), sigmaX=blur_kernel, sigmaY=blur_kernel)
        return blur_img
    
def get_pyramid_layer_num(img_shape):
    '''
    计算金字塔的层数
    '''
    min_size = min(img_shape)
    O = round(np.log(min_size) / np.log(2) -1)
    return int(O)

def get_blur_kernels_list(image_num_per_layer):
    '''
    获取每层每个图像对应的blur核
    '''
    k = 2**(1./ image_num_per_layer)
    image_num = image_num_per_layer + 3
    blur_kernels_list = np.zeros(image_num)
    blur_kernels_list[0] = sigma

    for index in range(1, image_num):
        pre = (k**(index-1))* sigma
        now = k * pre
        blur_kernels_list[index] = np.sqrt(now **2 - pre **2)

    return blur_kernels_list

def get_blur_image_list(image, layer_num, blur_kernel_list):
    '''
    对高斯金字塔图像进行整理
    '''
    blur_image_list = []
    
    for layer_index in range(layer_num):
        images_each_layer = [image]
        
        for blur_kernel in blur_kernel_list[1:]:
            image = cv2.GaussianBlur(image, (0,0), sigmaX= blur_kernel, sigmaY=blur_kernel)
            images_each_layer.append(image)
        blur_image_list.append(images_each_layer)
        
        #准备下一个层中的第一个图像,选取这一层图像的倒数第三个图像直接下采样
        image_2_next_layer = images_each_layer[-3]
        new_w =int(image_2_next_layer.shape[0]/2)
        new_h = int(image_2_next_layer.shape[1]/2)
        image = cv2.resize(image_2_next_layer, (new_h,new_w), interpolation=cv2.INTER_NEAREST)

    blur_image_list = np.array(blur_image_list, dtype=object)
    return blur_image_list

def get_DoG_image_list(image_list):
    '''
    高斯金字塔中相邻图像相减获得高斯差分图像
    '''
    DoG_image_list = []
    
    for images_each_layer in image_list:
        DoG_diff_list = []
        for pre_image,this_image in zip(images_each_layer,images_each_layer[1:]):
            DoG_diff_list.append(this_image-pre_image)
        DoG_image_list.append(DoG_diff_list)
    
    DoG_image_list = np.array(DoG_image_list, dtype=object)
    return DoG_image_list

def get_keypoints(blur_image_list, DoG_image_list, layer_num):
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
    
    threshold = np.floor( 0.5 * threshold_const / layer_num * 255)
    keypoints = []
    stride =3

    for layer_index, DoG_each_layer in enumerate(DoG_image_list):
        for image_index, (bottom_image, mid_image, top_image) in enumerate(zip(DoG_each_layer,DoG_each_layer[1:], DoG_each_layer[2:])):

            image_cube = np.stack([bottom_image,mid_image,top_image]).astype('float32') 

            W,H = image_cube.shape[1], image_cube.shape[2]

            for x in range(border, W-border):
                for y in range(border, H-border):

                    if is_keypoint(image_cube[:,x-1:x+2,y-1:y+2],threshold):
                        #离散的点不能求最优值，用拟合函数来计算
                        real_result = get_real_result(x,y,image_index+1,layer_index,layer_num,DoG_each_layer,image_cube.shape)
                        if real_result is not None:
                            keypoint, image_index_in_layer = real_result
                            o_keypoints = get_orientation(keypoint,layer_index,blur_image_list[layer_index][image_index_in_layer])
                            keypoints+=o_keypoints

    return keypoints
        

def get_orientation(keypoint, layer_index, image, radius=3, split_num=36, ratio=0.8, scale=1.5):
    '''
    获取图像像素点的方向
    '''
    o_keypoints = []
    shape = image.shape

    scale = scale * keypoint.size / np.float(2 ** (layer_index+1))
    radius = int(round(radius * scale))
    weight = -0.5 / (scale ** 2)
    #360度按照切分数目进行记录
    raw_h = np.zeros(split_num)
    smooth_h = np.zeros(split_num)
    

    for y_ in range(-radius, radius+1):
        y = int(round(keypoint.pt[1] / np.float(2 ** layer_index))) + y_
        
        #判断y和x是否在合适范围内
        if y > 0 and y < shape[0] - 1:
            for x_ in range(-radius, radius+1):
                x = int(round(keypoint.pt[0] / np.float(2 ** layer_index))) + x_
                
                if x > 0 and x < shape[1] - 1:
                    #计算梯度，此处和计算不同
                    dx = image[y,x+1] - image[y,x-1]
                    dy = image[y-1, x] - image[y+1, x]
                    
                    l = np.sqrt(dx**2 + dy**2)
                    
                    #笛卡尔坐标系转化找方向
                    GO = np.rad2deg(np.arctan2(dy, dx))
                    weight_tmp = np.exp(weight * (x_**2 + y_**2))
                    h_index = int(round(GO * split_num /360.))
                    raw_h[h_index % split_num] += weight_tmp* l

    for n in range(split_num):
        #根据H平滑公式进行计算
        smooth_h[n] = (6 * raw_h[n] + 4 * (raw_h[n - 1] + raw_h[(n + 1) % split_num]) + raw_h[n - 2] + raw_h[(n + 2) % split_num]) / 16.

    layer_max = np.max(smooth_h)
    roll_01 = smooth_h>np.roll(smooth_h,1) 
    roll_10 = smooth_h>np.roll(smooth_h,-1)
    tmp = roll_01 & roll_10
    # 选出指定维度的index
    layer_peaks = np.where(tmp)[0]
    
    # exit()
    for peak_index in layer_peaks:
        peak = smooth_h[peak_index]
        if peak>= ratio * layer_max:
            left = smooth_h[(peak_index -1) % split_num]
            right = smooth_h[(peak_index +1 ) % split_num]
            #计算连续最值点的位置
            mid_peak_index = (peak_index + 0.5 *(left - right)/(left -2*peak + right)) % split_num
            orientation = 360.- mid_peak_index *360. / split_num
            if np.abs(orientation - 360.) < 1e-7:
                orientation = 0
            
            o_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            o_keypoints.append(o_keypoint)
    return o_keypoints




def is_keypoint(image_cube, threshold):
    '''
    判断对应坐标的图像是否是关键极值点
    '''
    center_pixel = image_cube[1,1,1]
    abs_value = np.abs(center_pixel)
    if abs_value > threshold:
        if center_pixel > 0:
            #判断是不是最大点
            return np.all(center_pixel>=image_cube)

        elif center_pixel < 0:
            #判断是不是最小点
            return np.all(center_pixel<=image_cube)

    return False

def get_gradient_from_cube(cube):

    dx = 0.5 * (cube[1, 1, 2] - cube[1, 1, 0])
    dy = 0.5 * (cube[1, 2, 1] - cube[1, 0, 1])
    ds = 0.5 * (cube[2, 1, 1] - cube[0, 1, 1])
    return np.array([dx, dy, ds])

def get_hessian_from_cube(cube):
    center_pixel_value = cube[1, 1, 1]
    dxx = cube[1, 1, 2] - 2 * center_pixel_value + cube[1, 1, 0]
    dyy = cube[1, 2, 1] - 2 * center_pixel_value + cube[1, 0, 1]
    dss = cube[2, 1, 1] - 2 * center_pixel_value + cube[0, 1, 1]
    dxy = 0.25 * (cube[1, 2, 2] - cube[1, 2, 0] - cube[1, 0, 2] + cube[1, 0, 0])
    dxs = 0.25 * (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 0])
    dys = 0.25 * (cube[2, 2, 1] - cube[2, 0, 1] - cube[0, 2, 1] + cube[0, 0, 1])
    return np.array([[dxx, dxy, dxs], [dxy, dyy, dys],[dxs, dys, dss]])

def get_real_result(
                    x, y, 
                    image_index_in_layer, 
                    layer_index, 
                    image_num_per_layer,
                    cube,
                    shape,
                    ratio = 10,
                    try_times= 5):
    '''
    用函数对极值点进行判断
    离散的点无法确定极值点
    需要对主要的点判断偏移量

    x,y 是坐标
    '''
    out_flag = False

    shape = shape[1:]
    for t in range(try_times):
        bottom,mid,top = cube[image_index_in_layer-1:image_index_in_layer+2]
        pixel_cube = np.stack([
            bottom[x-1:x+2, y-1:y+2],
            mid[x-1:x+2, y-1:y+2],
            top[x-1:x+2, y-1:y+2],
        ])
        pixel_cube = pixel_cube.astype('float32') / 255.
        
        gradient = get_gradient_from_cube(pixel_cube)
        hessian  = get_hessian_from_cube(pixel_cube)
        #进行最小二乘线性拟合，找到合适点
        shift = - np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        #最优点的偏移不会偏移到其他像素点就自动结束
        if np.abs(shift[0])<0.5 and np.abs(shift[1])<0.5 and np.abs(shift[2])<0.5:
            break
        
        #这部分可能有bug
        x += int(round(shift[1]))
        y += int(round(shift[0]))
        image_index_in_layer += int(round(shift[2]))


        #查看是否极值点能落在原始图片区域里
        if x< border or x >= shape[0]-border or y<border or y>shape[1]-border or image_index_in_layer<1 or image_index_in_layer> image_num_per_layer:
            out_flag = True
            break

    #找不到最优解
    if out_flag:
        return None
    
    #优化到最低
    if t >= try_times-1:
        return None
    
    #解
    keypoint_tmp = pixel_cube[1,1,1]+ 0.5*np.dot(gradient, shift)
    
    if np.abs(keypoint_tmp) * image_num_per_layer >= threshold_const:
        xy_h = hessian[:2, :2]
        #求对角线元素之和
        xy_h_trace = np.trace(xy_h)
        xy_h_det = np.linalg.det(xy_h)
        
        # 对比检查
        if xy_h_det > 0 and ratio * (xy_h_trace ** 2) < ((ratio + 1) ** 2) * xy_h_det:
            keypoint = cv2.KeyPoint()
            
            keypoint.pt = ((y + shift[0]) * (2 ** layer_index), (x + shift[1]) * (2 ** layer_index))
            keypoint.octave = layer_index + image_index_in_layer * (2**8) + int(round((shift[2] + 0.5) * 255)) * (2**16)
            keypoint.size = sigma * (2 ** ((image_index_in_layer + shift[2]) / np.float32(image_num_per_layer))) * (2 ** (layer_index + 1))  
            keypoint.response = np.abs(keypoint_tmp)
            return keypoint, image_index_in_layer
    
    return None


def remove_duplication(keypoints):
    # 删除重复的点
    if len(keypoints) < 2:
        return keypoints
    
    keypoints.sort(key = cmp_to_key(compareKeypoints))
    good_keypoints = [keypoints[0]]
    
    for keypoint in keypoints[1:]:
        pre_keypoint = good_keypoints[-1]
        if pre_keypoint.pt[0] != keypoint.pt[0] or \
           pre_keypoint.pt[1] != keypoint.pt[1] or \
           pre_keypoint.size != keypoint.size or \
           pre_keypoint.angle != keypoint.angle:
            good_keypoints.append(keypoint)

    return good_keypoints

def change_image_size( keypoints):
    out_image_size = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave -1) & 255)
        out_image_size.append(keypoint)

    return out_image_size

def get_descriptors(keypoints, images, window_width=4, split_num=8, scale_m=3, max_value=0.2):
    descriptors=[]

    for keypoint in keypoints:

        octave, layer, scale = get_unpack_octave(keypoint)

        image = images[octave+1, layer]
        num_rows, num_cols = image.shape
        point = np.round(scale * np.array( keypoint.pt)).astype('int')
        split_pre_degree = split_num / 360.
        angle = 360. - keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))

        weighter = -0.5 / ((0.5 * window_width) ** 2)
        row_split_list = []
        col_split_list = []
        m_list = []
        o_split_list = []
        h_tensor = np.zeros((window_width+2, window_width+2, split_num))

        hist_width = scale_m * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * np.sqrt(2) * (window_width +1) * 0.5))
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

        for row in range(-half_width, half_width+1):
            for col in range(-half_width, half_width+1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width -0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width -0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows-1 and window_col > 0 and window_col < num_cols-1 :
                        dx = image[window_row ,window_col +1] - image[window_row, window_col-1]
                        dy = image[window_row-1, window_col] - image[window_row + 1, window_col]
                        g_m = np.sqrt(dx**2 + dy**2)
                        g_o = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weighter * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_split_list.append(row_bin)
                        col_split_list.append(col_bin)
                        m_list.append(weight * g_m)
                        o_split_list.append((g_o - angle) * split_pre_degree)
        
        # 逆三线性差值
        for row_v, col_v , m, o in zip(row_split_list, col_split_list, m_list, o_split_list):
            floor_row, floor_col, floor_o = np.floor([row_v, col_v, o]).astype(int)
            row_score, col_score, o_score = row_v - floor_row , col_v- floor_col, o-floor_o
            if  floor_o < 0:
                floor_o += split_num
            if floor_o  >= split_num:
                floor_o -= split_num

            c1 = m * row_score
            c0 = m * (1 - row_score)
            c11 = c1 * col_score
            c10 = c1 * (1 - col_score)
            c01 = c0 * col_score
            c00 = c0 * (1 - col_score)
            c111 = c11 * o_score
            c110 = c11 * (1 - o_score)
            c101 = c10 * o_score
            c100 = c10 * (1 - o_score)
            c011 = c01 * o_score
            c010 = c01 * (1 - o_score)
            c001 = c00 * o_score
            c000 = c00 * (1 - o_score)
            pre_list= [c000,c001,c010,c011,c100,c101,c110,c111]

            #除中心点之外的8个点进行处理
            h_tensor[floor_row + 1, floor_col + 1, floor_o] += pre_list[0]
            h_tensor[floor_row + 1, floor_col + 1, (floor_o + 1) % split_num] += pre_list[1]
            h_tensor[floor_row + 1, floor_col + 2, floor_o] += pre_list[2]
            h_tensor[floor_row + 1, floor_col + 2, (floor_o + 1) % split_num] += pre_list[3]
            h_tensor[floor_row + 2, floor_col + 1, floor_o] += pre_list[4]
            h_tensor[floor_row + 2, floor_col + 1, (floor_o + 1) % split_num] += pre_list[5]
            h_tensor[floor_row + 2, floor_col + 2, floor_o] += pre_list[6]
            h_tensor[floor_row + 2, floor_col + 2, (floor_o + 1) % split_num] += pre_list[7]
        
        #进行整理 
        d_vector = h_tensor[1:-1, 1:-1, :].flatten()
        threshold = np.linalg.norm(d_vector) * max_value
        d_vector[d_vector > threshold] = threshold
        d_vector /= max(np.linalg.norm(d_vector), 1e-7)
        d_vector = np.round(512 * d_vector)
        d_vector[d_vector<0] = 0
        d_vector[d_vector>255]  = 255
        descriptors.append(d_vector)
    descriptors = np.array(descriptors,dtype='float32')
    return descriptors

def get_unpack_octave(keypoint):
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1/np.float32(1 << octave) if octave >= 0 else np.float32( 1<< -octave)
    return octave, layer, scale 

def compareKeypoints(keypoint1, keypoint2):
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id


def run(image, image_num_per_layer):
    image = image.astype('float32')
    basic_image = get_basic_image(image,1.6,0.5)
    layer_num = get_pyramid_layer_num(basic_image.shape)
    blur_kernel_list = get_blur_kernels_list(image_num_per_layer)
    blur_image_list = get_blur_image_list(basic_image, layer_num, blur_kernel_list)
    DoG_image_list = get_DoG_image_list(blur_image_list)
    keypoints = get_keypoints(blur_image_list,DoG_image_list,3)
    keypoints = remove_duplication(keypoints)
    keypoints = change_image_size(keypoints)
    descriptors = get_descriptors(keypoints, blur_image_list)
    return keypoints, descriptors


def SFIT_func(image,layer_num):
    return run(image,layer_num)

