import os
import numpy as np 

def get_class(class_name):
    cls_ = 0
    if class_name == 'Pedestrian' or class_name == 'Person_sitting':
        cls_ = 1
    elif class_name == 'Cyclist':
        cls_ = 2
    elif class_name == 'Car' or class_name == 'Van':
        cls_ = 3
    elif class_name == 'Truck':
        cls_ = 4
    elif class_name == 'Tram':
        cls_ = 5
    else:
        cls_ = 0
    return cls_

def get_kitti_label(label_dir, coord='camera'):
    with open(label_dir, 'r') as fp:
        label_list = fp.readlines()
        result_list = []
        for i in range(len(label_list)):
            label = label_list[i].replace('\n', '')
            obj = label.split(' ')
            cls_ = float(get_class(obj[0]))
            if cls_ > 0:
                x = float(obj[11])
                y = float(obj[12])
                z = float(obj[13])
                l = float(obj[10])
                w = float(obj[9])
                h = float(obj[8])
                r = float(obj[14])
                if coord == 'velodyne':
                    x = float(obj[13]) + 0.27
                    y = - float(obj[11])
                    z = -0.08 - float(obj[12]) + (h/2)
                    r = -float(obj[14]) - np.pi/2
                    if r < -np.pi:
                        r = r + np.pi * 2
                label_obj = np.array([cls_, x, y, z, l, w, h, r]).reshape(1, 8)
                result_list.append(label_obj)
        if len(result_list) > 0:
            result = np.concatenate(result_list, axis=0)
        else: 
            result = None
    return result

def sort_label(label):
    x = label[:, 1]
    y = label[:, 2]
    z = label[:, 3]
    L = np.sqrt(x**2 + y**2 + z**2)
    idx = np.argsort(L)
    result = label[idx, :]
    result = result[::-1, :]
    return result

def test():
    label_dir = '/home/yifeihu/data/Kitti/object/training/label_2/000010.txt'
    objs = get_kitti_label(label_dir, coord='velodyne')
    print(objs)

#test()