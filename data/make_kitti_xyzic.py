import os
import numpy as np 
from PIL import Image

KITTI_DIR = '/home/yifeihu/data/Kitti/object/training'

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
    label_dir = KITTI_DIR + '/label_2/'
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
        data[:, 3] = 1
        p2, r0, velo2cam = get_calib(calibfile)
        Tmat = np.dot(p2, np.dot(r0, velo2cam)).T
        imdata = np.dot(data, Tmat)
        imdata[:, 0] = imdata[:, 0] / imdata[:, 2]
        imdata[:, 1] = imdata[:, 1] / imdata[:, 2]
        im = Image.open(imfile)
        width, height = im.size
        mask = (imdata[:, 0] > 0) & (imdata[:, 0] < width) & (imdata[:, 1] > 0) & (imdata[:, 1] < height)
        outdata = data[mask, :]
        outdata.tofile(outbinfile)
        

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