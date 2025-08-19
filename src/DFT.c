#include "DFT.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void dft(complex_array ft, float sample[], unsigned int samples) {
    for (unsigned int n = 0; n < ft.size; n++) {
        ft.imaginary[n] = 0.f;
        ft.real[n] = 0.f;
        for (unsigned int i = 0; i < samples; i++) {
            ft.imaginary[n] -= sample[i] * sin(2*PI*n*(float)i/ft.size);
            ft.real[n]      += sample[i] * cos(2*PI*n*(float)i/ft.size);
        }

    }
}

void reconstruct(float* x, complex_array X, unsigned int N) {
    for (unsigned int n = 0; n < N; n++) {
        double sum = 0.0;
        for (unsigned int k = 0; k < X.size; k++) {
            sum +=  X.real[k] * cos(2 * PI * k * n / X.size)
                  - X.imaginary[k] * sin(2 * PI * k * n / X.size); // minus
        }
        x[n] = (float)(sum / X.size);
    }
}


static void fft_recursive(float* real, float* imag, unsigned int n) {
    if (n <= 1) return;


    unsigned int half = n / 2;
    float* evenReal = (float*)malloc(half * sizeof(float));
    float* evenImag = (float*)malloc(half * sizeof(float));
    float* oddReal  = (float*)malloc(half * sizeof(float));
    float* oddImag  = (float*)malloc(half * sizeof(float));

    for (unsigned int i = 0; i < half; i++) {
        evenReal[i] = real[2*i];
        evenImag[i] = imag[2*i];
        oddReal[i]  = real[2*i+1];
        oddImag[i]  = imag[2*i+1];
    }

    fft_recursive(evenReal, evenImag, half);
    fft_recursive(oddReal, oddImag, half);

    for (unsigned int k = 0; k < half; k++) {
        float angle = -2.0f * (float)PI * k / n;
        float wr = cosf(angle);
        float wi = sinf(angle);

        float tr = wr * oddReal[k] - wi * oddImag[k];
        float ti = wr * oddImag[k] + wi * oddReal[k];

        real[k]     = evenReal[k] + tr;
        imag[k]     = evenImag[k] + ti;
        real[k+half]= evenReal[k] - tr;
        imag[k+half]= evenImag[k] - ti;
    }

    free(evenReal);
    free(evenImag);
    free(oddReal);
    free(oddImag);
}

void fft(complex_array ft, float sample[], unsigned int samples) {

    for (unsigned int i = 0; i < samples; i++) {
        ft.real[i] = sample[i];
        ft.imaginary[i] = 0.0f;
    }

    fft_recursive(ft.real, ft.imaginary, samples);
}

void ifft(complex_array ft, float* x, unsigned int N) {
    for (unsigned int i = 0; i < N; i++) {
        ft.imaginary[i] = -ft.imaginary[i];
    }


    fft(ft, ft.real, N);

    for (unsigned int i = 0; i < N; i++) {
        ft.real[i]      =  ft.real[i] / N;
        ft.imaginary[i] = -ft.imaginary[i] / N;
        x[i] = ft.real[i];
    }
}