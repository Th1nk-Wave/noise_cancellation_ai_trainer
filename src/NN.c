#include "NN.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ucontext.h>

#if NN_DEBUG_PRINT
#if defined(__linux__)
// https://linux.die.net/man/3/malloc_usable_size
#include <malloc.h>
size_t malloc_size(const void *p) {
    return malloc_usable_size((void*)p);
}
#elif defined(__APPLE__)
// https://www.unix.com/man-page/osx/3/malloc_size/
#include <malloc/malloc.h>
size_t malloc_size(const void *p) {
    return malloc_size(p);
}
#elif defined(_WIN32)
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/msize
#include <malloc.h>
size_t malloc_size(const void *p) {
    return _msize((void *)p);
}
#else
#error "system not recognised when determining malloc_size debug function, you may be able to fix this by disabling the NN_DEBUG_PRINT flag or adding your own definitions"
#endif
#endif




NN_network* NN_network_init(unsigned int* neurons_per_layer,
                            unsigned int layers) {
    NN_network* net = (NN_network*)malloc(sizeof(NN_network));
    net->layers = layers;
    net->neurons_per_layer = malloc(sizeof(unsigned int) * layers);
    for (unsigned int layer = 0; layer < layers; layer++) {
        net->neurons_per_layer[layer] = neurons_per_layer[layer];
    }

    net->weights = malloc(sizeof(float**) * (layers - 1));
    net->bias = malloc(sizeof(float*) * (layers - 1));

    for (unsigned int layer = 0; layer < layers - 1; layer++) {
        unsigned int in_n = neurons_per_layer[layer];
        unsigned int out_n = neurons_per_layer[layer + 1];

        net->weights[layer] = malloc(sizeof(float*) * in_n);
        net->bias[layer] = malloc(sizeof(float) * out_n);

        for (unsigned int i = 0; i < in_n; i++) {
            net->weights[layer][i] = malloc(sizeof(float) * out_n);
            #if NN_INIT_ZERO
                for (unsigned int j = 0; j < out_n; j++) {
                    net->weights[layer][i][j] = 0.0f;
                }
            #endif
        }

        #if NN_INIT_ZERO
            for (unsigned int j = 0; j < out_n; j++) {
                net->bias[layer][j] = 0.0f;
            }
        #endif
    }

    net->in = malloc(sizeof(float) * neurons_per_layer[0]);
    net->out = malloc(sizeof(float*) * layers);
    for (unsigned int layer = 0; layer < layers; layer++) {
        net->out[layer] = malloc(sizeof(float) * neurons_per_layer[layer]);
        #if NN_INIT_ZERO
            for (unsigned int i = 0; i < neurons_per_layer[layer]; i++) {
                net->out[layer][i] = 0.0f;
            }
        #endif
    }

    #if NN_DEBUG_PRINT
    size_t neuron_layer_alloc = sizeof(unsigned int) * layers;
    size_t in_alloc = sizeof(float) * neurons_per_layer[0];

    size_t weights_alloc = sizeof(float**) * (layers - 1);
    size_t bias_alloc = sizeof(float*) * (layers - 1);

    for (unsigned int layer = 0; layer < layers - 1; layer++) {
        weights_alloc += sizeof(float*) * (neurons_per_layer[layer]); // each weight row
        weights_alloc += sizeof(float*) * 1; // extra pointer for bias connection?
        bias_alloc += sizeof(float) * (neurons_per_layer[layer + 1]);

        weights_alloc += sizeof(float) * (neurons_per_layer[layer + 1]) * (neurons_per_layer[layer] + 1);
    }

    size_t out_alloc = sizeof(float*) * layers;
    for (unsigned int layer = 0; layer < layers; layer++) {
        out_alloc += sizeof(float) * neurons_per_layer[layer];
    }

    size_t total = neuron_layer_alloc + in_alloc + weights_alloc + bias_alloc + out_alloc;

    printf("allocated %.6f MiB of memory for neural network\n", total / (1024.0 * 1024.0));
    printf("    L neuron_layer_info: %.6f MiB (%.2f%%)\n", neuron_layer_alloc / (1024.0 * 1024.0), 100.0 * neuron_layer_alloc / total);
    printf("    L in_neurons: %.6f MiB (%.2f%%)\n", in_alloc / (1024.0 * 1024.0), 100.0 * in_alloc / total);
    printf("    L out_neurons: %.6f MiB (%.2f%%)\n", out_alloc / (1024.0 * 1024.0), 100.0 * out_alloc / total);
    printf("    L weights: %.6f MiB (%.2f%%)\n", weights_alloc / (1024.0 * 1024.0), 100.0 * weights_alloc / total);
    printf("    L bias: %.6f MiB (%.2f%%)\n", bias_alloc / (1024.0 * 1024.0), 100.0 * bias_alloc / total);
    #endif

    return net;
}

void NN_network_free(NN_network *net) {
    #if NN_DEBUG_PRINT
        size_t total_freed = 0;
    #endif

    for (unsigned int layer = 0; layer < net->layers - 1; layer++) {
        unsigned int in_n = net->neurons_per_layer[layer];

        for (unsigned int i = 0; i < in_n; i++) {
            #if NN_DEBUG_PRINT
                total_freed += malloc_size(net->weights[layer][i]);
            #endif
            free(net->weights[layer][i]);
        }
        #if NN_DEBUG_PRINT
            total_freed += malloc_size(net->weights[layer]);
            total_freed += malloc_size(net->bias[layer]);
        #endif
        free(net->weights[layer]);
        free(net->bias[layer]);
    }

    #if NN_DEBUG_PRINT
        total_freed += malloc_size(net->weights);
        total_freed += malloc_size(net->bias);
        printf("freed network weights and biases: %.6f MiB\n", (float)total_freed / (1024 * 1024));
    #endif
    free(net->weights);
    free(net->bias);

    for (unsigned int i = 0; i < net->layers; i++) {
        free(net->out[i]);
    }
    free(net->in);
    free(net->out);

    free(net->neurons_per_layer);
    free(net);

    #if NN_MEMORY_TRIM_AFTER_FREE
    #ifdef __linux__
        malloc_trim(0);
    #endif
    #endif
}

void NN_network_randomise_xaivier(NN_network *net, float weight_min, float weight_max) {
    for (unsigned int l = 0; l < net->layers - 1; l++) {
        unsigned int in_n = net->neurons_per_layer[l];
        unsigned int out_n = net->neurons_per_layer[l + 1];
        float limit = sqrtf(6.0f / (float)(in_n + out_n));
        for (unsigned int i = 0; i < in_n; i++) {
            for (unsigned int j = 0; j < out_n; j++) {
                net->weights[l][i][j] = random_float_range(-limit, limit);
            }
        }
        for (unsigned int j = 0; j < out_n; j++) {
            net->bias[l][j] = 0.0f;
        }
    }
}

NN_trainer* NN_trainer_init(NN_network* network, NN_learning_settings* learning_settings, NN_use_settings* use_settings, char *device_name) {
    unsigned int _size = strlen(device_name) + 1;

    NN_trainer* trainer = (NN_trainer*)malloc(sizeof(NN_trainer));
    trainer->processor.device_name = malloc(_size);
    for (unsigned int i = 0; i < _size; i++) {
        trainer->processor.device_name[i] = device_name[i];
    }

    trainer->learning_settings = learning_settings;
    trainer->processor.network = network;
    trainer->processor.settings = use_settings;

    trainer->m_weights = NULL;
    trainer->v_weights = NULL;
    trainer->m_bias = NULL;
    trainer->v_bias = NULL;
    trainer->adam_t = 0;

    // malloc weight+bias gradients if using batching
    if (learning_settings->use_batching) {
        unsigned int layers = network->layers;
        unsigned int* neurons_per_layer = network->neurons_per_layer;

        trainer->grad_weights = malloc(sizeof(float**) * (layers-1));
        trainer->grad_bias = malloc(sizeof(float*) * (layers-1));

        for (unsigned int layer = 0; layer < layers - 1; layer++) {
            unsigned int in_n = neurons_per_layer[layer];
            unsigned int out_n = neurons_per_layer[layer + 1];
    
            trainer->grad_weights[layer] = malloc(sizeof(float*) * in_n);
            trainer->grad_bias[layer] = calloc(out_n, sizeof(float));
    
            for (unsigned int i = 0; i < in_n; i++) {
                trainer->grad_weights[layer][i] = calloc(out_n, sizeof(float));
            }
        }
    } else {
        trainer->grad_weights = NULL;
        trainer->grad_bias = NULL;
    }

    // malloc a and v accumulators if using ADAM
    if (trainer->learning_settings->optimizer == ADAM || trainer->learning_settings->optimizer == ADAMW) {
        unsigned int layers = network->layers;
        unsigned int* neurons_per_layer = network->neurons_per_layer;

        trainer->m_weights = malloc(sizeof(float**) * (layers - 1));
        trainer->v_weights = malloc(sizeof(float**) * (layers - 1));
        trainer->m_bias = malloc(sizeof(float*) * (layers - 1));
        trainer->v_bias = malloc(sizeof(float*) * (layers - 1));

        for (unsigned int layer = 0; layer < layers - 1; layer++) {
            unsigned int in_n = neurons_per_layer[layer];
            unsigned int out_n = neurons_per_layer[layer + 1];

            trainer->m_weights[layer] = malloc(sizeof(float*) * in_n);
            trainer->v_weights[layer] = malloc(sizeof(float*) * in_n);

            trainer->m_bias[layer] = calloc(out_n, sizeof(float));
            trainer->v_bias[layer] = calloc(out_n, sizeof(float));

            for (unsigned int i = 0; i < in_n; i++) {
                trainer->m_weights[layer][i] = calloc(out_n, sizeof(float));
                trainer->v_weights[layer][i] = calloc(out_n, sizeof(float));
            }
        }

        trainer->adam_t = 1;
    }

    return trainer;
}

void NN_trainer_free(NN_trainer* trainer) {

    // free gradient buffers if using batching
    if (trainer->learning_settings->use_batching) {
        unsigned int layers = trainer->processor.network->layers;
        unsigned int* neurons_per_layer = trainer->processor.network->neurons_per_layer;
        
        #if NN_DEBUG_PRINT
            size_t total_freed = 0;
        #endif

        for (unsigned int layer = 0; layer < layers - 1; layer++) {
            unsigned int in_n = neurons_per_layer[layer];
    
            for (unsigned int i = 0; i < in_n; i++) {
                #if NN_DEBUG_PRINT
                    total_freed += malloc_size(trainer->grad_weights[layer][i]);
                #endif
                free(trainer->grad_weights[layer][i]);
            }
            #if NN_DEBUG_PRINT
                total_freed += malloc_size(trainer->grad_weights[layer]);
                total_freed += malloc_size(trainer->grad_bias[layer]);
            #endif
            free(trainer->grad_weights[layer]);
            free(trainer->grad_bias[layer]);
        }
    
        #if NN_DEBUG_PRINT
            total_freed += malloc_size(trainer->grad_weights);
            total_freed += malloc_size(trainer->grad_bias);
            printf("freed training gradient buffers: %.6f MiB\n", (float)total_freed / (1024 * 1024));
        #endif
        free(trainer->grad_weights);
        free(trainer->grad_bias);
    }

    // free adam accumulators if using ADAM
    if (trainer->learning_settings->optimizer == ADAM || trainer->learning_settings->optimizer == ADAMW) {
        unsigned int layers = trainer->processor.network->layers;
        unsigned int* neurons_per_layer = trainer->processor.network->neurons_per_layer;

        for (unsigned int layer = 0; layer < layers - 1; layer++) {
            unsigned int in_n = neurons_per_layer[layer];
            for (unsigned int i = 0; i < in_n; i++) {
                free(trainer->m_weights[layer][i]);
                free(trainer->v_weights[layer][i]);
            }
            free(trainer->m_weights[layer]);
            free(trainer->v_weights[layer]);
            free(trainer->m_bias[layer]);
            free(trainer->v_bias[layer]);
        }
        free(trainer->m_weights);
        free(trainer->v_weights);
        free(trainer->m_bias);
        free(trainer->v_bias);
    }

    free(trainer->processor.device_name);
    free(trainer);
    
}

NN_processor* NN_processor_init(NN_network* network, NN_use_settings* settings, char* device_name) {
    unsigned int _size = strlen(device_name) + 1;

    NN_processor* processor = (NN_processor*)malloc(sizeof(NN_processor));
    processor->device_name = malloc(_size);
    for (unsigned int i = 0; i < _size; i++) {
        processor->device_name[i] = device_name[i];
    }

    processor->settings = settings;
    processor->network = network;

    return processor;
}

void NN_processor_free(NN_processor* processor) {
    free(processor->device_name);
    free(processor);
}

void NN_processor_process(NN_processor* processor, float* in, float* out) {
    NN_network* net = processor->network;
    unsigned int layers = net->layers;

    for (unsigned int i = 0; i < net->neurons_per_layer[0]; i++) {
        net->in[i] = in[i];
        net->out[0][i] = in[i];
    }

    // forward pass
    for (unsigned int layer = 1; layer < layers; layer++) {
        for (unsigned int j = 0; j < net->neurons_per_layer[layer]; j++) {
            float sum = net->bias[layer - 1][j];
            for (unsigned int i = 0; i < net->neurons_per_layer[layer - 1]; i++) {
                sum += net->out[layer - 1][i] * net->weights[layer - 1][i][j];
            }

            // apply activation
            switch (processor->settings->activation) {
                case RELU:
                    net->out[layer][j] = sum > 0 ? sum : 0;
                    break;
                case SIGMOID:
                    net->out[layer][j] = 1.0f / (1.0f + expf(-sum));
                    break;
                case TANH:
                    net->out[layer][j] = tanhf(sum);
                    break;
                case SOFTMAX:
                    // handled at output layer
                    net->out[layer][j] = sum; 
                    break;
            }
        }

        if (processor->settings->activation == SOFTMAX && layer == layers - 1) {
            float max = net->out[layer][0];
            for (unsigned int i = 1; i < net->neurons_per_layer[layer]; i++)
                if (net->out[layer][i] > max) max = net->out[layer][i];

            float sum = 0.0f;
            for (unsigned int i = 0; i < net->neurons_per_layer[layer]; i++) {
                net->out[layer][i] = expf(net->out[layer][i] - max);
                sum += net->out[layer][i];
            }
            for (unsigned int i = 0; i < net->neurons_per_layer[layer]; i++) {
                net->out[layer][i] /= sum;
            }
        }
    }

    unsigned int last = layers - 1;
    for (unsigned int i = 0; i < net->neurons_per_layer[last]; i++) {
        out[i] = net->out[last][i];
    }
}

void NN_trainer_train(NN_trainer* trainer, float* in, float* desired_out) {
    NN_network* net = trainer->processor.network;
    NN_use_settings* settings = trainer->processor.settings;
    float lr = trainer->learning_settings->learning_rate;
    unsigned int layers = net->layers;

    // run forward pass
    NN_processor_process(&trainer->processor, in, net->out[layers - 1]);

    // allocate deltas
    float** deltas = malloc(sizeof(float*) * layers);
    for (unsigned int l = 0; l < layers; l++) {
        deltas[l] = calloc(net->neurons_per_layer[l], sizeof(float));
    }

    // compute output layer error
    unsigned int out_layer = layers - 1;
    for (unsigned int i = 0; i < net->neurons_per_layer[out_layer]; i++) {
        float out = net->out[out_layer][i];
        float err = out - desired_out[i];
        float deriv = 1.0f;
        switch (settings->activation) {
            case SIGMOID:
                deriv = out * (1.0f - out);
                break;
            case TANH:
                deriv = 1.0f - out * out;
                break;
            case RELU:
                deriv = (out > 0) ? 1.0f : 0.0f;
                break;
            case SOFTMAX:
                deriv = 1.0f; // usually combined with cross-entropy loss
                break;
        }
        deltas[out_layer][i] = err * deriv;
    }

    // backpropagation for hidden layers
    for (int l = layers - 2; l >= 0; l--) {
        for (unsigned int i = 0; i < net->neurons_per_layer[l]; i++) {
            float out = net->out[l][i];
            float delta_sum = 0.0f;
            for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
                delta_sum += net->weights[l][i][j] * deltas[l + 1][j];
            }
            float deriv = 1.0f;
            switch (settings->activation) {
                case SIGMOID: deriv = out * (1.0f - out); break;
                case TANH: deriv = 1.0f - out * out; break;
                case RELU: deriv = (out > 0) ? 1.0f : 0.0f; break;
                default: break;
            }
            deltas[l][i] = delta_sum * deriv;
        }
    }

    // update weights and biases
    for (unsigned int l = 0; l < layers - 1; l++) {
        for (unsigned int i = 0; i < net->neurons_per_layer[l]; i++) {
            for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
                net->weights[l][i][j] -= lr * deltas[l + 1][j] * net->out[l][i];
            }
        }
        for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
            net->bias[l][j] -= lr * deltas[l + 1][j];
        }
    }

    // free deltas
    for (unsigned int l = 0; l < layers; l++) {
        free(deltas[l]);
    }
    free(deltas);
}

void NN_trainer_accumulate(NN_trainer *trainer, float *input, float *target) {
    if (!trainer->learning_settings->use_batching) { fprintf(stderr, "ERROR: can't accumulate to gradient buffers because they have not been allocated\n    FIX: enable use_batching in learning_settings, this will roughly double memory usage though as a result");}
    NN_network* net = trainer->processor.network;
    NN_use_settings* settings = trainer->processor.settings;
    float lr = trainer->learning_settings->learning_rate;
    unsigned int layers = net->layers;

    // run forward pass
    NN_processor_process(&trainer->processor, input, net->out[layers - 1]);

    // allocate deltas
    float** deltas = malloc(sizeof(float*) * layers);
    for (unsigned int l = 0; l < layers; l++) {
        deltas[l] = calloc(net->neurons_per_layer[l], sizeof(float));
    }

    // compute output layer error
    unsigned int out_layer = layers - 1;
    for (unsigned int i = 0; i < net->neurons_per_layer[out_layer]; i++) {
        float out = net->out[out_layer][i];
        float err = out - target[i];
        float deriv = 1.0f;
        switch (settings->activation) {
            case SIGMOID:
                deriv = out * (1.0f - out);
                break;
            case TANH:
                deriv = 1.0f - out * out;
                break;
            case RELU:
                deriv = (out > 0) ? 1.0f : 0.0f;
                break;
            case SOFTMAX:
                deriv = 1.0f; // usually combined with cross-entropy loss
                break;
        }
        deltas[out_layer][i] = err * deriv;
    }

    // backpropagation for hidden layers
    for (int l = layers - 2; l >= 0; l--) {
        for (unsigned int i = 0; i < net->neurons_per_layer[l]; i++) {
            float out = net->out[l][i];
            float delta_sum = 0.0f;
            for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
                delta_sum += net->weights[l][i][j] * deltas[l + 1][j];
            }
            float deriv = 1.0f;
            switch (settings->activation) {
                case SIGMOID: deriv = out * (1.0f - out); break;
                case TANH: deriv = 1.0f - out * out; break;
                case RELU: deriv = (out > 0) ? 1.0f : 0.0f; break;
                default: break;
            }
            deltas[l][i] = delta_sum * deriv;
        }
    }

    // add updates to gradient buffer
    for (unsigned int l = 0; l < layers - 1; l++) {
        for (unsigned int i = 0; i < net->neurons_per_layer[l]; i++) {
            for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
                trainer->grad_weights[l][i][j] += deltas[l + 1][j] * net->out[l][i];
            }
        }
        for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
            trainer->grad_bias[l][j] += deltas[l + 1][j];
        }
    }

    // free deltas
    for (unsigned int l = 0; l < layers; l++) {
        free(deltas[l]);
    }
    free(deltas);
}

void NN_trainer_apply(NN_trainer *trainer, unsigned int batch_size) {
    NN_network* net = trainer->processor.network;
    unsigned int layers = net->layers;
    NN_learning_settings* ls = trainer->learning_settings;
    float lr = ls->learning_rate;

    switch (trainer->learning_settings->optimizer) {
        case ADAM: 
        case ADAMW: {
            /* ADAM equation:
            * m = beta1*m + (1-beta1)*g
            * v = beta2*v + (1-beta2)*g^2
            * m_hat = m / (1 - beta1^t)
            * v_hat = v / (1 - beta2^t)
            * w -= lr * m_hat / (sqrt(v_hat) + eps)
            *
            * where g is the average gradient over the batch (grad / batch_size).
            */
            const float beta1 = (ls->adam_beta1 > 0.0f) ? ls->adam_beta1 : 0.9f;
            const float beta2 = (ls->adam_beta2 > 0.0f) ? ls->adam_beta2 : 0.999f;
            const float eps   = (ls->adam_epsilon > 0.0f) ? ls->adam_epsilon : 1e-8f;
            const float wd    = (ls->weight_decay > 0.0f) ? ls->weight_decay : 0.01f;

            trainer->adam_t++;

            float one_minus_beta1_t = 1.0f - powf(beta1, (float)trainer->adam_t);
            float one_minus_beta2_t = 1.0f - powf(beta2, (float)trainer->adam_t);

            for (unsigned int l = 0; l < layers - 1; l++) {
                unsigned int in_n  = net->neurons_per_layer[l];
                unsigned int out_n = net->neurons_per_layer[l + 1];

                // weights
                for (unsigned int i = 0; i < in_n; i++) {
                    for (unsigned int j = 0; j < out_n; j++) {
                        float g = trainer->grad_weights[l][i][j] / (float)batch_size;

                        float clip_value = 1.0f;
                        if (g > clip_value) g = clip_value;
                        else if (g < -clip_value) g = -clip_value;

                        // ADAM update
                        float m = trainer->m_weights[l][i][j];
                        float v = trainer->v_weights[l][i][j];
                        m = beta1 * m + (1.0f - beta1) * g;
                        v = beta2 * v + (1.0f - beta2) * (g * g);
                        trainer->m_weights[l][i][j] = m;
                        trainer->v_weights[l][i][j] = v;

                        float m_hat = m / one_minus_beta1_t;
                        float v_hat = v / one_minus_beta2_t;

                        // ADAMW decoupled weight decay
                        if (ls->optimizer == ADAMW) {
                            net->weights[l][i][j] -= lr * wd * net->weights[l][i][j];
                        }

                        net->weights[l][i][j] -= lr * (m_hat / (sqrtf(v_hat) + eps));
                        trainer->grad_weights[l][i][j] = 0.0f;
                    }
                }

                // bias
                for (unsigned int j = 0; j < out_n; j++) {
                    float g = trainer->grad_bias[l][j] / (float)batch_size;

                    float clip_value = 1.0f;
                    if (g > clip_value) g = clip_value;
                    else if (g < -clip_value) g = -clip_value;

                    float m = trainer->m_bias[l][j];
                    float v = trainer->v_bias[l][j];
                    m = beta1 * m + (1.0f - beta1) * g;
                    v = beta2 * v + (1.0f - beta2) * (g * g);
                    trainer->m_bias[l][j] = m;
                    trainer->v_bias[l][j] = v;

                    float m_hat = m / one_minus_beta1_t;
                    float v_hat = v / one_minus_beta2_t;

                    if (ls->optimizer == ADAMW) {
                        net->bias[l][j] -= lr * wd * net->bias[l][j];
                    }

                    net->bias[l][j] -= lr * (m_hat / (sqrtf(v_hat) + eps));
                    trainer->grad_bias[l][j] = 0.0f;
                }
            }
            break;
        }
        default: {
            // update weights and biases
            for (unsigned int l = 0; l < layers - 1; l++) {
                for (unsigned int i = 0; i < net->neurons_per_layer[l]; i++) {
                    for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
                        net->weights[l][i][j] -= lr * (trainer->grad_weights[l][i][j] / batch_size);
                        trainer->grad_weights[l][i][j] = 0;
                    }
                }
                for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
                    net->bias[l][j] -= lr * (trainer->grad_bias[l][j] / batch_size);
                    trainer->grad_bias[l][j] = 0;
                }
            }
            break;
        }
    }
}

float NN_trainer_loss(NN_trainer* trainer, float* desired) {
    NN_network* net = trainer->processor.network;
    float loss = 0.0f;
    unsigned int last = net->layers - 1;
    unsigned int out_n = net->neurons_per_layer[last];
    for (unsigned int i = 0; i < out_n; i++) {
        float diff = net->out[last][i] - desired[i];
        loss += diff * diff;
    }
    return loss / out_n;
}

int NN_network_save_to_file(NN_network *net, char *filepath) {
    FILE *f = fopen(filepath, "wb");
    if (!f) return -1;

    uint16_t version = NN_FILE_VERSION;
    uint8_t activation_id = 0;

    if (fwrite(&version, sizeof(uint16_t), 1, f) != 1) goto error;                                      // [2 byte] version
    if (fwrite(&activation_id, sizeof(uint8_t), 1, f) != 1) goto error;                                 // [1 byte] activation id 
    if (fwrite(&net->layers, sizeof(uint32_t), 1, f) != 1) goto error;                                  // [4 byte] layers
    if (fwrite(net->neurons_per_layer, sizeof(uint32_t), net->layers, f) != net->layers) goto error;    // [4 * layers byte] neurons_per_layer

    // weights
    for (unsigned int l = 0; l < net->layers - 1; l++) {
        unsigned int in_count  = net->neurons_per_layer[l];
        unsigned int out_count = net->neurons_per_layer[l + 1];
        for (unsigned int i = 0; i < in_count; i++) {
            if (fwrite(net->weights[l][i], sizeof(float), out_count, f) != out_count) goto error;
        }
    }

    // biases
    for (unsigned int l = 0; l < net->layers - 1; l++) {
        unsigned int out_count = net->neurons_per_layer[l + 1];
        if (fwrite(net->bias[l], sizeof(float), out_count, f) != out_count) goto error;
    }

    fclose(f);
    return 0;

error:
    fclose(f);
    return -1;
}

NN_network* NN_network_init_from_file(char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return NULL;

    uint16_t version;
    uint8_t activation_id;
    uint32_t layers;

    // version check
    if (fread(&version, sizeof(uint16_t), 1, f) != 1) goto error;       // [2 byte] version
    if (version != NN_FILE_VERSION) {
        fprintf(stderr, "unsupported .net file version: %u\n", version);
        goto error;
    }

    if (fread(&activation_id, sizeof(uint8_t), 1, f) != 1) goto error;  // [1 byte] activation id
    if (fread(&layers, sizeof(uint32_t), 1, f) != 1) goto error;        // [4 byte] layers

    // neruons_per_layer
    unsigned int* neurons_per_layer = malloc(sizeof(uint32_t) * layers);
    if (!neurons_per_layer) goto error;
    if (fread(neurons_per_layer, sizeof(uint32_t), layers, f) != layers) goto error;

    // malloc network
    NN_network* net = NN_network_init(neurons_per_layer, layers);
    if (!net) goto error;

    // free neurons_per_layer (no longer needed)
    free(neurons_per_layer);

    // weights
    for (unsigned int l = 0; l < layers - 1; l++) {
        unsigned int in_count  = net->neurons_per_layer[l];
        unsigned int out_count = net->neurons_per_layer[l + 1];
        for (unsigned int i = 0; i < in_count; i++) {
            if (fread(net->weights[l][i], sizeof(float), out_count, f) != out_count) goto error;
        }
    }

    // bias
    for (unsigned int l = 0; l < layers - 1; l++) {
        unsigned int out_count = net->neurons_per_layer[l + 1];
        if (fread(net->bias[l], sizeof(float), out_count, f) != out_count) goto error;
    }

    fclose(f);
    return net;

error:
    fclose(f);
    return NULL;
}

int NN_network_load_from_file(NN_network *net, char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) goto no_file;

    uint16_t version;
    uint8_t activation_id;
    uint32_t layers;

    // version check
    if (fread(&version, sizeof(uint16_t), 1, f) != 1) goto error;       // [2 byte] version
    if (version != NN_FILE_VERSION) {
        fprintf(stderr, ".net file version missmatch, file:%u, current:%u\n", version, NN_FILE_VERSION);
        goto error;
    }

    if (fread(&activation_id, sizeof(uint8_t), 1, f) != 1) goto error;  // [1 byte] activation id
    if (fread(&layers, sizeof(uint32_t), 1, f) != 1) goto error;        // [4 byte] layers
    // shape check
    if (net->layers != layers) goto wrong_shape;


    // neruons_per_layer
    unsigned int* neurons_per_layer = malloc(sizeof(uint32_t) * layers);
    if (!neurons_per_layer) goto error;
    if (fread(neurons_per_layer, sizeof(uint32_t), layers, f) != layers) goto error;

    // shape check 
    if (memcmp(net->neurons_per_layer, neurons_per_layer, layers*sizeof(unsigned int))!=0) {free(neurons_per_layer); goto wrong_shape; }
    free(neurons_per_layer);

    // weights
    for (unsigned int l = 0; l < layers - 1; l++) {
        unsigned int in_count  = net->neurons_per_layer[l];
        unsigned int out_count = net->neurons_per_layer[l + 1];
        for (unsigned int i = 0; i < in_count; i++) {
            if (fread(net->weights[l][i], sizeof(float), out_count, f) != out_count) goto error;
        }
    }

    // bias
    for (unsigned int l = 0; l < layers - 1; l++) {
        unsigned int out_count = net->neurons_per_layer[l + 1];
        if (fread(net->bias[l], sizeof(float), out_count, f) != out_count) goto error;
    }

    printf("succesfully loaded network.\n");
    fclose(f);
    return 0;
no_file:
    printf("could not find file: %s\n",filepath);
    return -3;
error:
    fclose(f);
    return -1;
wrong_shape:
    fclose(f);
    fprintf(stderr, "missmatched network shape when loading from file: %s\n",filepath);
    return -2;
}