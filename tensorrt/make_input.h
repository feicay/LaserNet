#ifndef __MAKE_INPUT__
#define __MAKE_INPUT__

void xyzic_to_image(float* xyzic, int num, float h_start, float h_end, float v_start, float v_end, float dh, float dv, float* im, float* im_cls);
void make_xyzi_to_image(float* xyzic, int num, float* im, float* im_cls);

#endif