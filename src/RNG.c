#include "RNG.h"

// RNG - Marsaglia's xor32
static unsigned int seed = 0x12345678;
unsigned int random_uint() {
    seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}

float random_float() {
    return random_uint() * 2.3283064365387e-10f;
}

float random_float_range(float min, float max) {
    return (random_float()*(max-min))+min;
}

unsigned int random_uint_range(unsigned int min, unsigned int max) {
    return (random_float()*(max-min))+min;
}