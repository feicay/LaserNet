#include "make_input.h"
#include <math.h>
#include <string.h>

#define PI (3.14159)

void xyzic_to_image(float* xyzic, int num, float h_start, float h_end, float v_start, float v_end, float dh, float dv, float* im, float* im_cls)
{
    int H = int((v_end - v_start + 0.01)/dv);
    int W = int((h_end - h_start + 0.01)/dh);
    float x,y,z,intensity,yaw,v_angle;
    int c;
    int w,h;
    memset(im, 0, W*H*3);
    memset(im_cls, 0, W*H);
    for(int i=0; i<num; i++)
    {
        x = xyzic[i*5 + 0];
        y = xyzic[i*5 + 1];
        z = xyzic[i*5 + 2];
        intensity = xyzic[i*5 + 3];
        c = int(xyzic[i*5 + 4] + 0.01);
        yaw = atan2(y, x);
        v_angle = atan2(z, sqrt(x*x + y*y));
        w = int((yaw - h_start)/dh + 0.5);
        h = int((v_angle - v_end)/dv + 0.5);
        if((w>=0) && (w<W) && (h>=0) && (h<H))
        {
            //range  reflict  height
            im[H*W*0 + h*W + w] = sqrt(x*x + y*y + z*z);
            im[H*W*1 + h*W + w] = intensity;
            im[H*W*1 + h*W + w] = z;
            im_cls[h*W + w] = c;
        }
    }
}

void make_xyzi_to_image(float* xyzic, int num, float* im, float* im_cls)
{
    int W = 400;
    int H = 200;
    float dh = 0.225;
    float dv = 0.2;
    xyzic_to_image(xyzic, num, 0, 90, -30, 10, dh, dv, im, im_cls);
    xyzic_to_image(xyzic, num, 90, 180, -30, 10, dh, dv, (im+W*H*3), (im_cls+W*H));
    xyzic_to_image(xyzic, num, -180, -90, -30, 10, dh, dv, (im+2*W*H*3), (im_cls+2*W*H));
    xyzic_to_image(xyzic, num, -90, 0, -30, 10, dh, dv, (im+3*W*H*3), (im_cls+3*W*H));
}