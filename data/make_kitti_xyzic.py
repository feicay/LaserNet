import os
import numpy as np 
from PIL import Image
from get_label import get_kitti_label, sort_label
import cv2

KITTI_DIR = '/home/adas/data/Kitti/object/testing'
color = np.array([[0, 0, 0],
                  [0, 0, 250],
                  [0, 250, 250],
                  [0, 250, 0],
                  [250, 250, 0],
                  [250, 0, 0],
                  [250, 0, 250]])

def get_calib(calibfile):
    with open(calibfile, 'r') as fp:
        text = fp.readlines()
        p2 = text[2].replace('\n', '').split(': ')[1]
        r0 = text[4].replace('\n', '').split(': ')[1]
        velo2cam = text[5].replace('\n', '').split(': ')[1]
    p2 = np.array(p2.split(' ')).reshape(3, 4).astype(np.float32)
    r0 = np.array(r0.split(' ')).reshape(3, 3).astype(np.float32)
    velo2cam = np.array(velo2cam.split(' ')).reshape(3, 4).astype(np.float32)
    zero1 = np.zeros((3,1))
    const = np.array([[0, 0, 0, 1]])
    r0 = np.concatenate((r0, zero1), axis=1)
    r0 = np.concatenate((r0, const), axis=0)
    velo2cam = np.concatenate((velo2cam, const), axis=0)
    return p2, r0, velo2cam

def make_velodyne_reduce():
    bin_dir = KITTI_DIR + '/velodyne/'
    im_dir =  KITTI_DIR + '/image_2/'
    calib_dir = KITTI_DIR + '/calib/'
    bin_reduce_dir = KITTI_DIR + '/velodyne_reduce/'
    binlist = os.listdir(bin_dir)
    for i in range(len(binlist)):
        print(i, binlist[i])
        binfile = binlist[i]
        calibfile = calib_dir + binfile.replace('bin', 'txt')
        imfile = im_dir + binfile.replace('bin', 'png')
        rawbinfile = bin_dir + binfile
        outbinfile = bin_reduce_dir + binfile
        data = np.fromfile(rawbinfile, dtype=np.float32).reshape(-1, 4)
        num, _ = data.shape
        xyz = data[:, 0:3]
        xyz1 = np.concatenate((xyz, np.ones((num, 1))), axis=1)
        p2, r0, velo2cam = get_calib(calibfile)
        Tmat = np.dot(p2, np.dot(r0, velo2cam)).T
        imdata = np.dot(xyz1, Tmat)
        imdata[:, 0] = imdata[:, 0] / imdata[:, 2]
        imdata[:, 1] = imdata[:, 1] / imdata[:, 2]
        im = Image.open(imfile)
        width, height = im.size
        mask = (imdata[:, 0] > 0) & (imdata[:, 0] < width) & (imdata[:, 1] > 0) & (imdata[:, 1] < height) & (data[:, 0] > 0)
        outdata = data[mask, :]
        outdata.tofile(outbinfile)

def get_xyzic(data, label, h_offset=1.73):
    if label is None:
        return data
    num, _ = data.shape
    n, _ = label.shape
    c = np.zeros((num, 1), dtype=np.float32)
    for i in range(n):
        obj = label[i, :]
        print(obj)
        cls_ , x, y, z, l, w, h, r = obj
        delta_x = data[:, 0] - np.ones(num)*x
        delta_y = data[:, 1] - np.ones(num)*y
        delta_z = data[:, 2] - np.ones(num)*z 
        theta = np.arctan2(delta_y, delta_x) - np.ones(num)*r
        L = np.sqrt(delta_x*delta_x + delta_y*delta_y)
        delta_w = L * np.sin(theta)
        delta_l = L * np.cos(theta)
        mask = (delta_w > (-w/2)) & (delta_w < (w/2)) & (delta_l > (-l/2)) & (delta_l < (l/2)) & (delta_z > (-h/2)) & (delta_z < (h/2))
        c[mask, :] = cls_
    xyzic = np.concatenate((data, c), axis=1)
    return xyzic

def make_xyzic_image(xyzic, yaw_start=-45, yaw_end=45, v_start=-30, v_end=10, d_yaw=0.2, d_v=0.2):
    num, _ = xyzic.shape
    width = int((yaw_end - yaw_start) / d_yaw + 0.01)
    height = int((v_end - v_start) / d_v + 0.01)
    im_ref = np.zeros((height, width), dtype=np.float32)
    im_height = np.zeros((height, width), dtype=np.float32)
    im_cls = np.zeros((height, width, 3), dtype=np.float32)
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
    im_ref[j, i] = xyzic[:, 3]
    im_height[j, i] = xyzic[:, 2] + 1.73
    im_cls[j, i] = color[xyzic[:, 4].astype(np.int32), :]
    im_range[j, i] = L
    return im_cls, im_ref, im_height, im_range


def make_kitti_xyzic():
    bin_dir = KITTI_DIR + '/velodyne_reduce/'
    bin_xyzic_dir = KITTI_DIR + '/velodyne_xyzic/'
    label_dir = KITTI_DIR + '/label_2/'
    binlist = os.listdir(bin_dir)
    for i in range(len(binlist)):
    #for i in range(1):
        print(i, binlist[i])
        binfile = binlist[i]  
        labelfile = label_dir + binfile.replace('bin', 'txt')
        rawbinfile = bin_dir + binfile
        outbinfile = bin_xyzic_dir + binfile
        label = get_kitti_label(labelfile, 'velodyne')
        label = sort_label(label)
        data = np.fromfile(rawbinfile, dtype=np.float32).reshape(-1, 4)
        #data[:, 2] = data[:, 2]
        xyzic = get_xyzic(data, label)
        im_cls, im_ref, im_height, im_range = make_xyzic_image(xyzic)
        # im1 = Image.fromarray(im_cls.astype('uint8')).convert('RGB')
        # im1.show()
        # im2 = Image.fromarray((im_ref*255).astype('uint8')).convert('RGB')
        # im2.show()
        # im3 = Image.fromarray((im_height*255).astype('uint8')).convert('RGB')
        # im3.show()
        # im4 = Image.fromarray((im_range/100*255).astype('uint8')).convert('RGB')
        # im4.show()
        xyzic.tofile(outbinfile)


def test():
    calibfile = '/home/yifeihu/data/Kitti/object/training/calib/000000.txt'
    binfile = '/home/yifeihu/data/Kitti/object/training/velodyne/000000.bin'
    imfile = '/home/yifeihu/data/Kitti/object/training/image_2/000000.png'
    p2, r0, velo2cam = get_calib(calibfile)
    print(p2)
    print(r0)
    print(velo2cam)
    T = np.dot(p2, np.dot(r0, velo2cam))
    print(T)
    data = np.fromfile(binfile, dtype=np.float32).reshape(-1, 4)
    data[:, 3] = 1
    print(data.shape)
    imdata = np.dot(data, T.T)
    imdata[:, 0] = imdata[:, 0] / imdata[:, 2]
    imdata[:, 1] = imdata[:, 1] / imdata[:, 2]
    print(imdata[:10, :])
    im = Image.open(imfile)
    width, height = im.size
    print(width, height)
    mask = (imdata[:, 0] > 0) & (imdata[:, 0] < width) & (imdata[:, 1] > 0) & (imdata[:, 1] < height)
    outdata = data[mask, :]
    print(outdata.shape)

#make_velodyne_reduce()
#test()
make_kitti_xyzic()