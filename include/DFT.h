#ifndef __DFT_H_
#define __DFT_H_

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979323846

typedef struct {
    float* real;
    float* imaginary;
    unsigned int size;
} complex_array;

void dft(complex_array ft, float sample[], unsigned int samples);
void fft(complex_array ft, float sample[], unsigned int samples);
void reconstruct(float* reconstructed, complex_array transform, unsigned int size);
void ifft(complex_array ft, float* x, unsigned int N);

#endif