#ifndef __WAV_UTIL_H_
#define __WAV_UTIL_H_

#include <stdint.h>
#include <stdio.h>

//util
void write_wav_header(FILE* f, uint32_t data_size, uint16_t channels, uint32_t sample_rate, uint16_t bits_per_sample);

#endif