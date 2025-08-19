#include "RNG.h"
#include <iso646.h>
#include <math.h>
#include <stdbool.h>
#define DR_WAV_IMPLEMENTATION

#include <stdio.h>
#include <portaudio.h>
#include <pa_ringbuffer.h>
#include <dr_wav.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <raylib.h>

#include "NN.h"
#include "DFT.h"

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))


#define RANDOM_INIT_MAX 0.1
#define RANDOM_INIT_MIN -0.1

#define LEARNING_RATE 0.0001
#define LEARNING_TEST_SPLIT 0.7
#define PARAMETERS (1<<8)
#define BATCH_SIZE 800


#define RING_BUFFER_SIZE (1 << 17) // Must be power of 2
#define ZERO_OUTPUT_BUFFER 1

typedef struct {
    NN_trainer* trainer;
    float* input;
    float* target;
} trainer_thread_args;

typedef struct{
    complex_array* ft;
    float* sample;
    unsigned int samples;
} dft_thread_args;

void* trainer_thread_func(void* __args) {
    trainer_thread_args* args = (trainer_thread_args*)__args;
    NN_trainer_accumulate(args->trainer, args->input,args->target);
    return NULL;
}

void* dft_thread_func(void* __args) {
    dft_thread_args* args = (dft_thread_args*)__args;
    dft(*args->ft,args->sample,args->samples);
}


void rms_normalize_complex_array(complex_array *data, float target_rms) {
    if (!data || data->size == 0 || target_rms <= 0.0f) return;

    double sum_sq = 0.0;
    unsigned int len = data->size;

    for (unsigned int i = 0; i < len; i++) {
        double re = (double)data->real[i];
        double im = (double)data->imaginary[i];
        sum_sq += re * re + im * im;
    }

    double rms = sqrt(sum_sq / (double)len);
    if (rms < 1e-12) return; // avoid division by zero

    double scale = target_rms / rms;

    for (unsigned int i = 0; i < len; i++) {
        data->real[i]      = (float)(data->real[i] * scale);
        data->imaginary[i] = (float)(data->imaginary[i] * scale);

        // clamp values
        //if (data->real[i] > 1) {data->real[i]=1;}
        //if (data->real[i] < -1) {data->real[i]=-1;}
        //if (data->imaginary[i] > 1) {data->imaginary[i]=1;}
        //if (data->imaginary[i] < -1) {data->imaginary[i]=-1;}
    }
}

float rms_compute_complex_array(const complex_array *data) {
    double sum_sq = 0.0;
    for (unsigned int i = 0; i < data->size; i++) {
        double re = (double)data->real[i];
        double im = (double)data->imaginary[i];
        sum_sq += re * re + im * im;
    }
    return (float)sqrt(sum_sq / (double)data->size);
}

void rms_denormalize_complex_array(complex_array *data, float original_rms, float target_rms) {
    if (target_rms <= 0.0f) return;
    float scale = original_rms / target_rms;
    for (unsigned int i = 0; i < data->size; i++) {
        data->real[i] *= scale;
        data->imaginary[i] *= scale;
    }
}

void draw_wave(float* points, unsigned int size, Color col) {
    for (unsigned int p1 = 0; p1 < size-2; p1++) {
        float p1_data = points[p1];
        float p2_data = points[p1+1];
        DrawLine(p1, 200-p1_data*100, p1+1, 200-p2_data*100, col);
    }
}

int main(int argc, char** argv) {
    // usage check
    if (argc < 4) {
        printf("Usage: %s <clean_speak.wav> <background_noise.wav> <foreground_noise.wav>\n", argv[0]);
        return 1;
    }
    

    // init wav streams
    drwav tts_speech;
    drwav background_noise;
    drwav foreground_noise;
    if (!drwav_init_file(&tts_speech, argv[1], NULL)) {fprintf(stderr, "failed to open %s\n",argv[1]);return 1;}
    if (!drwav_init_file(&background_noise, argv[2], NULL)) {fprintf(stderr, "failed to open %s\n",argv[2]);return 1;}
    if (!drwav_init_file(&foreground_noise, argv[3], NULL)) {fprintf(stderr, "failed to open %s\n",argv[3]);return 1;}

    // settings
    NN_learning_settings* learning_settings = (NN_learning_settings*)malloc(sizeof(NN_learning_settings));
    NN_use_settings* use_settings = (NN_use_settings*)malloc(sizeof(NN_use_settings));
    learning_settings->learning_rate = LEARNING_RATE;
    learning_settings->adam_beta1 = 0; // 0 to let lib automatically choose it
    learning_settings->adam_beta2 = 0;
    learning_settings->adam_epsilon = 0;
    learning_settings->weight_decay = 0;
    use_settings->activation = TANH;
    learning_settings->optimizer = ADAMW;
    learning_settings->use_batching = true;
    use_settings->device_type = CPU;

    // init
    unsigned int neurons_per_layer[5] = {PARAMETERS,(PARAMETERS<<1),(PARAMETERS<<1),PARAMETERS,PARAMETERS};
    NN_network* net_real = NN_network_init(neurons_per_layer, 5);
    NN_network* net_imaginary = NN_network_init(neurons_per_layer, 5);
    NN_trainer* trainer_real = NN_trainer_init(net_real, learning_settings, use_settings, "cpu1");
    NN_trainer* trainer_imaginary = NN_trainer_init(net_imaginary, learning_settings, use_settings, "cpu1");
    
    if (NN_network_load_from_file(net_real, "real_latest.net")==-3) {
        // if no network to load from exists, just randomise
        NN_network_randomise_xaivier(net_real, RANDOM_INIT_MIN, RANDOM_INIT_MAX);
    }

    if (NN_network_load_from_file(net_imaginary, "imaginary_latest.net")==-3) {
        NN_network_randomise_xaivier(net_imaginary, RANDOM_INIT_MIN, RANDOM_INIT_MAX);
    }

    // malloc audio buffers
    float* clean = malloc(PARAMETERS * sizeof(float) * tts_speech.channels);
    float* background = malloc(PARAMETERS * sizeof(float));
    float* foreground = malloc(PARAMETERS * sizeof(float));
    float* noisey = malloc(PARAMETERS * sizeof(float));

    // alloc complex array
    float* noisy_imag = malloc(PARAMETERS * sizeof(float));
    float* noisy_real = malloc(PARAMETERS * sizeof(float));
    complex_array noisey_ft = {
        .imaginary = noisy_imag,
        .real = noisy_real,
        .size = PARAMETERS
    };

    float* clean_imag = malloc(PARAMETERS * sizeof(float));
    float* clean_real = malloc(PARAMETERS * sizeof(float));
    complex_array clean_ft = {
        .imaginary = clean_imag,
        .real = clean_real,
        .size = PARAMETERS
    };

    pthread_t dft_thread;
    pthread_t real_thread;

    trainer_thread_args trainer_thread_arg = {
        .trainer=trainer_real,
        .input=noisy_real,
        .target=clean_real
    };

    dft_thread_args dft_thread_arg = {
        .ft = &noisey_ft,
        .sample = noisey,
        .samples = PARAMETERS,
    };

    // init display window
    InitWindow(PARAMETERS, 400, "live comparisson");

    


    // start training
    unsigned int epoch = 0;
    float loss = 0;
    for (unsigned int i = 0; i < 400000; i++) {
        loss = 0;
        for (unsigned int batch = 0; batch < BATCH_SIZE; batch++) {

            // seek
            drwav_seek_to_pcm_frame(&tts_speech, random_uint_range(0,tts_speech.totalPCMFrameCount-(PARAMETERS+1)));
            drwav_seek_to_pcm_frame(&background_noise, random_uint_range(0,background_noise.totalPCMFrameCount-(PARAMETERS+1)));
            drwav_seek_to_pcm_frame(&foreground_noise, random_uint_range(0,foreground_noise.totalPCMFrameCount-(PARAMETERS+1)));

            // read
            drwav_read_pcm_frames_f32(&tts_speech, PARAMETERS, clean);
            drwav_read_pcm_frames_f32(&foreground_noise, PARAMETERS, foreground);
            drwav_read_pcm_frames_f32(&background_noise, PARAMETERS, background);

            // mix signals
            for (unsigned int i = 0; i < PARAMETERS; i++) {
                noisey[i] = (clean[i] + background[i] + foreground[i]) / 3;
            }

            // compute fft
            fft(noisey_ft, noisey, PARAMETERS);
            //pthread_create(&dft_thread, NULL, dft_thread_func, (void*)&dft_thread_arg);
            fft(clean_ft, clean, PARAMETERS);
            //pthread_join(dft_thread, NULL);
            
            // normalise network inputs
            float original_rms_clean = rms_compute_complex_array(&clean_ft);
            rms_normalize_complex_array(&clean_ft, 0.25);
            rms_normalize_complex_array(&noisey_ft, 0.25);

            
            // accumulate network
            //pthread_create(&real_thread, NULL, trainer_thread_func, (void*)&trainer_thread_arg);
            NN_trainer_accumulate(trainer_real, noisy_real,clean_real);
            NN_trainer_accumulate(trainer_imaginary, noisy_imag,clean_imag);
            //pthread_join(real_thread, NULL);

            // get loss
            loss += NN_trainer_loss(trainer_real, clean_real) * 0.5;
            loss += NN_trainer_loss(trainer_imaginary, clean_imag) * 0.5;

            // draw original waves
            //BeginDrawing();
            //ClearBackground(DARKGRAY);
            //draw_wave(clean_imag, PARAMETERS, WHITE);
            //draw_wave(clean_real, PARAMETERS, BLUE);

            // draw network output
            //draw_wave(trainer_imaginary->processor.network->out[trainer_imaginary->processor.network->layers-1], PARAMETERS, RED);
            //draw_wave(trainer_real->processor.network->out[trainer_real->processor.network->layers-1], PARAMETERS, GREEN);
            //EndDrawing();
            
        }
        printf("epoch %i, loss: %f\n", epoch, loss/BATCH_SIZE);
        NN_trainer_apply(trainer_real, BATCH_SIZE);
        NN_trainer_apply(trainer_imaginary, BATCH_SIZE);
        NN_network_save_to_file(net_imaginary, "imaginary_latest.net");
        NN_network_save_to_file(net_real, "real_latest.net");
        epoch++;
    }

    NN_network_save_to_file(net_imaginary, "imaginary_latest.net");
    NN_network_save_to_file(net_real, "real_latest.net");


    
    


    // clean up
    free(clean);
    free(background);
    free(foreground);
    free(noisey);

    free(noisy_imag);
    free(noisy_real);

    free(clean_imag);
    free(clean_real);

    NN_trainer_free(trainer_real);
    NN_trainer_free(trainer_imaginary);

    NN_network_free(net_imaginary);
    NN_network_free(net_real);

    drwav_uninit(&tts_speech);
    drwav_uninit(&background_noise);
    drwav_uninit(&foreground_noise);
    return 0;
}