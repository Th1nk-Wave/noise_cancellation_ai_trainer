#ifndef __NN_H_
#define __NN_H_

#include "RNG.h"

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define NN_FILE_VERSION 0
#define NN_VERSION "0.1.0"

#define NN_DEBUG_PRINT 1
#define NN_INIT_ZERO 1
#define NN_MEMORY_TRIM_AFTER_FREE 1

typedef enum {
    AUTO,
    GPU,
    GPU_CUDA,
    CPU,
    TPU,
} NN_device;

typedef enum {
    RELU,
    SIGMOID,
    SOFTMAX,
    TANH,
} NN_activation_function;

typedef enum {
    ADAM,
    GRADIENT_DESCENT,
    STOCHASTIC_GRADIENT_DESCENT,
    MINI_BATCH_GRADIENT_DESCENT,
} NN_optimizer;

typedef enum {
    MSE,
} NN_loss_function;


typedef struct {
    float learning_rate;
    NN_optimizer optimizer;
    bool use_batching;

    // adam
    float adam_beta1;
    float adam_beta2;
    float adam_epsilon;

} NN_learning_settings;

typedef struct {
    NN_activation_function activation;
    NN_device device_type;
} NN_use_settings;


typedef struct {
    float* in;
    float** out;

    unsigned int* neurons_per_layer;
    unsigned int layers;

    float*** weights;
    float** bias;
} NN_network;

typedef struct {
    NN_use_settings* settings;
    char* device_name;
    NN_network* network;
} NN_processor;

typedef struct {
    NN_learning_settings* learning_settings;
    NN_processor processor;
    float ***grad_weights; // same shape as weights
    float **grad_bias;     // same shape as bias

    // adam state (only allocated when optimizer == ADAM)
    float ***m_weights;
    float ***v_weights;
    float **m_bias;
    float **v_bias;
    unsigned int adam_t;   // time step for bias correction
} NN_trainer;


// init functions
NN_network* NN_network_init(unsigned int* neurons_per_layer,
                            unsigned int layers);
NN_network* NN_network_init_from_file(char* filepath);
NN_trainer* NN_trainer_init(NN_network* network, NN_learning_settings* learn_settings, NN_use_settings* use_settings, char* device_name);
NN_processor* NN_processor_init(NN_network* network, NN_use_settings* settings, char* device_name);

// free functions
void NN_network_free(NN_network* network);
void NN_trainer_free(NN_trainer* trainer);
void NN_processor_free(NN_processor* processor);

// use functions
void NN_trainer_train(NN_trainer* trainer, float* in, float* desired_out);
void NN_trainer_accumulate(NN_trainer *trainer, float *input, float *target);
void NN_trainer_apply(NN_trainer *trainer, unsigned int batch_size);
void NN_processor_process(NN_processor* processor, float* in, float* out);

// utility functions
void NN_network_randomise_xaivier(NN_network* network, float weight_min, float weight_max);
float NN_trainer_loss(NN_trainer* trainer, float* desired);
int NN_network_load_from_file(NN_network* network, char* filepath);
int NN_network_save_to_file(NN_network* network, char* filepath);


#endif