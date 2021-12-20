import cv2
import numpy as np
from numpy.core.defchararray import index
from numpy.core.fromnumeric import shape
from SIFT import  SIFT_func

class Concat_Helper:
    def __init__(self):
        self.go = 1
    
    def get_image_list(self, path_list):
        self.gary_images = [cv2.resize(cv2.imread(path,0),(300, 400)) for path in path_list]
        self.images = [cv2.imread(path) for path in path_list]

    def run(self):
        img = self.get_new_image(0,1)
            
        return 1

    
    def get_new_image(self,index1, index2):
        FLANN_INDEX_KDTREE = 0
        MIN_MATCH_COUNT = 10
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        img1 = self.gary_images[index1]
        img2 = self.gary_images[index2]
        kp1, des1=SIFT_func(img2,3)
        kp2, des2 =SIFT_func(img1,3)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.array([ kp1[m.queryIdx].pt for m in good],'int32').reshape(-1,  2)
            dst_pts = np.array([ kp2[m.trainIdx].pt for m in good],'int32').reshape(-1,  2)

            M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)[0]
            out_xy = self.get_x_y(M,img2.shape)

            max_w = int(max(np.array(out_xy)[:,0]))
            min_x = int(min(np.array(out_xy)[:,0]))

            img1 = self.images[index1]
            img2 = self.images[index2]
            cv2.imshow('1',img1)
            cv2.imshow('2',img2)
            w, h = img1 .shape[0:2]
            warp_img = cv2.warpPerspective(img2,M,(max_w,400))

            processWidth = img1.shape[1]-min_x
            for w_ in range(w):
                for h_ in range(h):
                    if warp_img[w_,h_,:].all() != 0 and w_>=min_x:
                        rate = (processWidth - (w_ - min_x)) / processWidth
                    else:
                        rate = 1
                    warp_img[w_,h_,:] = img1[w_,h_,:]*rate + (1-rate)*warp_img[w_,h_,:]

            cv2.imshow('test',warp_img)
            cv2.waitKey()
            exit()
    
    def get_x_y(self,H,shape):
        out = []
        v2 = np.array([0,0,1]).T
        v1 = np.array([0,0,0]).T
        v1 = np.matmul(H ,v2)
        out.append([v1[0]/v1[2], v1[1]/v1[2]])

        v2 = np.array([0,shape[0],1]).T
        v1 = np.matmul(H ,v2)
        out.append([v1[0]/v1[2], v1[1]/v1[2]])

        v2 = np.array([shape[1],0,1]).T
        v1 = np.matmul(H ,v2)
        out.append([v1[0]/v1[2], v1[1]/v1[2]])
        
        v2 = np.array([shape[1],shape[0],1]).T
        v1 = np.matmul(H ,v2)
        out.append([v1[0]/v1[2], v1[1]/v1[2]])
        return out

if __name__ =='__main__':
    concat = Concat_Helper()
    concat.get_image_list(['11.png','22.png'])
    concat.run()
