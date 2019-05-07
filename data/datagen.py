import torch
import torch.utils.data as data
import numpy as np
import random
import cv2

def make_xyzic_image(xyzic, yaw_start=-45, yaw_end=45, v_start=-30, v_end=10, d_yaw=0.225, d_v=0.2):
    num, _ = xyzic.shape
    width = int((yaw_end - yaw_start) / d_yaw + 0.01)
    height = int((v_end - v_start) / d_v + 0.01)
    im_ref = np.zeros((height, width), dtype=np.float32)
    im_height = np.zeros((height, width), dtype=np.float32)
    im_cls = np.zeros((height, width), dtype=np.float32)
    im_range = np.zeros((height, width), dtype=np.float32)
    x = xyzic[:, 0]
    y = xyzic[:, 1]
    z = xyzic[:, 2]
    L = np.sqrt(x**2 + y**2 + z**2)
    yaw = np.arctan2(y, x)
    v_angel = np.arctan2(z, np.sqrt(x*x + y*y))
    i = ((yaw_end - yaw*180/np.pi) / d_yaw).astype(np.int32)
    j = ((v_end - v_angel*180/np.pi) / d_v).astype(np.int32)
    mask = (i > -1) & (i < width) & (j > -1) & (j < height)
    i = i[mask]
    j = j[mask]
    xyzic = xyzic[mask, :]
    im_cls[j, i] = (xyzic[:, 4] + 0.01)
    im_ref[j, i] = xyzic[:, 3]
    im_height[j, i] = xyzic[:, 2] + 1.73
    im_range[j, i] = L[mask]
    #convert to -1~1
    im_range = im_range / 100
    im_height = im_height / 10
    im = np.stack((im_range, im_ref, im_height))
    return im, im_cls.astype(np.int32)

class Lidar_xyzic_dataset(data.Dataset):
    def __init__(self, filelist, train=1):
        with open(filelist, 'r') as fp:
            self.filelist = fp.readlines()
            self.len = len(self.filelist)
            fp.close()
        self.train = train

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        binfile = self.filelist[index].replace('\n', '')
        data = np.fromfile(binfile, dtype=np.float32).reshape(-1, 5)
        if self.train:
            yaw_start = (random.random() - 0.5) * 360
            yaw_end = yaw_start + 90
            im, im_cls = make_xyzic_image(data, yaw_start=yaw_start, yaw_end=yaw_end)
        else:
            im, im_cls = make_xyzic_image(data)
        image = torch.from_numpy(im)
        truth = torch.from_numpy(im_cls)
        truth = truth.long().unsqueeze(0)
        return image, truth

'''category
‘DontCare’: 0
‘cyclist’: 1 
‘tricycle’: 2 
‘smallMot’: 3 
‘bigMot’: 4 
‘pedestrian’: 5 
‘crowds’: 6 
‘unknown’: 7
'''

def test():
    color = np.array([[0, 0, 0],
                  [0, 0, 250],
                  [0, 250, 250],
                  [0, 250, 0],
                  [250, 250, 0],
                  [250, 0, 0],
                  [250, 0, 250],
                  [150, 150, 150]])
    datalist = '/home/adas/data/pytorch_ws/LaserNet/trainlist.txt'
    dataset = Lidar_xyzic_dataset(datalist)
    image, truth = dataset.__getitem__(1000)
    print(image.size(), truth.size())
    _, h, w = image.size()
    image = image.permute(1, 2, 0).numpy()
    truth = truth.numpy()
    cv2.imshow('image', image)
    #cv2.waitKey(0)
    im_cls = np.zeros((h, w, 3), dtype=np.uint8)
    im_cls[:,:] = color[truth[:,:]]
    cv2.imshow('cls', im_cls)
    cv2.waitKey(0)

#test()