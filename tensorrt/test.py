import cv2
import numpy as np 

def test_truth():
    i = 537
    fname = 'bin/%06d.bin'%i
    im = np.fromfile(fname, dtype=np.float32)
    im = im.reshape(400,320)
    cv2.imshow("truth", im)
    cv2.waitKey(0)

test_truth()