import os
import numpy as np 
import pandas as pd 
from PIL import Image
import random

def get_points(pts_file, intensity_file, cat_file=None):
    pts = pd.read_csv(pts_file, header=None)
    points_loc = np.array(pts, dtype=np.float32)
    N, _ = points_loc.shape
    Inten = pd.read_csv(intensity_file, header=None)
    points_i = np.array(Inten, dtype=np.float32).reshape(-1, 1)
    if cat_file is not None:
        cat = pd.read_csv(cat_file, header=None)
        points_cat = np.array(cat, dtype=np.float32).reshape(-1, 1)
        points_xyzi = np.concatenate([points_loc, points_i, points_cat], axis=1)
    else:
        points_cat = np.zeros((N, 1))
        points_xyzi = np.concatenate([points_loc, points_i, points_cat], axis=1)
    #return points_filter(points_xyzi)
    return points_xyzi

def make_ali_lidar_xyzic():
    ali_lidar_dir = '/raid/alibaba-lidar/training/'
    int_dir = ali_lidar_dir + 'intensity/'
    pts_dir = ali_lidar_dir + 'pts/'
    cat_dir = ali_lidar_dir + 'category/'
    out_dir = ali_lidar_dir + 'xyzic/'
    filelist = os.listdir(pts_dir)
    for i in range(len(filelist)):
        print(i)
        filename = filelist[i]
        ptsfile = pts_dir + filename
        intfile = int_dir + filename
        catfile = cat_dir + filename
        outfile = out_dir + filename.replace('csv', 'bin')
        xyzic = get_points(ptsfile, intfile, cat_file=catfile)
        xyzic = xyzic.astype(np.float32)
        xyzic.tofile(outfile)

def make_train_list():
    fdir = '/raid/alibaba-lidar/training/xyzic'
    filelist = os.listdir(fdir)
    with open('trainlist.txt', 'w') as fp1, open('validlist.txt', 'w') as fp2:
        for i in range(len(filelist)):
            filename = fdir + '/' + filelist[i] + '\n'
            a = random.random()
            if a < 0.9:
                fp1.write(filename)
            else:
                fp2.write(filename)

    
#make_ali_lidar_xyzic()
#make_train_list()