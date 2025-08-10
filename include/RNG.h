#ifndef __RNG_H_
#define __RNG_H_

#define FLOAT_MAX 3.402823466e+38F

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define clamp(a,x,b) (((x)<(a))?(a):((b)<(x))?(b):(x))
//#define abs(x)   (((x)<(0))?(-x):(x))

unsigned int random_uint();
float random_float();
float random_float_range(float min, float max);

#endif