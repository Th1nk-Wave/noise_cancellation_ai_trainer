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

#include "NN.h"
#include "DFT.h"

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))


#define RANDOM_INIT_MAX 0.1
#define RANDOM_INIT_MIN -0.1

#define LEARNING_RATE 0.01
#define LEARNING_TEST_SPLIT 0.7
#define PARAMETERS (1<<9)
#define BATCH_SIZE 800


#define RING_BUFFER_SIZE (1 << 16) // Must be power of 2
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

static PaUtilRingBuffer ringBuffer;
static void* ringBufferData;

typedef struct {
    int numChannels;
    PaSampleFormat format;
} AudioState;

bool Patipping = false;

static int paCallback(const void *inputBuffer, void *outputBuffer,
                      unsigned long framesPerBuffer,
                      const PaStreamCallbackTimeInfo* timeInfo,
                      PaStreamCallbackFlags statusFlags,
                      void *userData)
{
    AudioState* audio = (AudioState*)userData;

    size_t samplesNeeded = framesPerBuffer * audio->numChannels;
    size_t samplesAvailable = PaUtil_GetRingBufferReadAvailable(&ringBuffer);

    if (PaUtil_GetRingBufferWriteAvailable(&ringBuffer) == 0) {
        Patipping = true;
    }

    #if ZERO_OUTPUT_BUFFER
        // Zero out in case of underrun (means instead of hearing rancid sounds you just hear nothing)
        memset(outputBuffer, 0, samplesNeeded*sizeof(float));
    #endif

    if (Patipping) {
        

        size_t samplesToRead = (samplesAvailable < samplesNeeded) ? samplesAvailable : samplesNeeded;

        PaUtil_ReadRingBuffer(&ringBuffer, outputBuffer, samplesToRead);

        if (PaUtil_GetRingBufferReadAvailable(&ringBuffer) == 0) {
            Patipping = false;
        }
    }

    

    return paContinue;
}

void* readerThreadFunc(void* _wav) {
    drwav* wav = (drwav*)_wav;
    printf("\n\n\n\nwav channels: %i\nsample rate: %i\n",wav->channels,wav->sampleRate);

    const size_t chunkSize = 4096;
    unsigned char temp[chunkSize];

    while (wav->bytesRemaining>0) {
        size_t space = PaUtil_GetRingBufferWriteAvailable(&ringBuffer);
        size_t toWrite = MIN3(space, chunkSize, wav->bytesRemaining);

        if (toWrite > 0) {
            size_t bytesRead = drwav_read_raw(wav, toWrite, temp);
            PaUtil_WriteRingBuffer(&ringBuffer, temp, bytesRead);
        } else {
            Pa_Sleep(10); // it's full already, give it some space.
        }
    }

    return NULL;
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


    // init portAudio
    printf("initialising audio interface...\n");
    AudioState audio = {
        .numChannels = 1,
        .format = paFloat32,
    };

    ringBufferData = malloc(RING_BUFFER_SIZE*sizeof(float));
    PaUtil_InitializeRingBuffer(&ringBuffer, sizeof(float), RING_BUFFER_SIZE, ringBufferData);
    Pa_Initialize();

    // select audio output device
    int numDevices = Pa_GetDeviceCount();
    const PaDeviceInfo* deviceInfo;

    int selectedDevice = 0;

    for (int i = 0; i < numDevices; ++i) {
        deviceInfo = Pa_GetDeviceInfo(i);
        printf("[%d] %s (%s)\n", i, deviceInfo->name, Pa_GetHostApiInfo(deviceInfo->hostApi)->name);
        if (strcmp(deviceInfo->name, "Default Sink")==0) {selectedDevice = i;}
    }

    // check if device is suitable
    deviceInfo = Pa_GetDeviceInfo(selectedDevice);
    const PaHostApiInfo* host = Pa_GetHostApiInfo(deviceInfo->hostApi);
    if (deviceInfo->maxOutputChannels < 1) {
        fprintf(stderr, "No suitable output device found\nenter device number you want to use: ");
        scanf("%i",&selectedDevice);
        deviceInfo = Pa_GetDeviceInfo(selectedDevice);
        host = Pa_GetHostApiInfo(deviceInfo->hostApi);
    }
    printf("Using device [%d]: %s (%s)\n", selectedDevice, deviceInfo->name, host->name);

    // configure device stream
    PaStreamParameters outputParams = {
        .device = selectedDevice,
        .channelCount = 1,
        .sampleFormat = paFloat32,
        .suggestedLatency = deviceInfo->defaultLowOutputLatency,
        .hostApiSpecificStreamInfo = NULL
    };

    PaStream* stream;
    Pa_OpenStream(&stream, NULL, &outputParams, tts_speech.sampleRate,
                  paFramesPerBufferUnspecified, paClipOff, paCallback, &audio);

    PaError err = Pa_StartStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "Pa_StartStream failed: %s\n", Pa_GetErrorText(err));
        return 1;
    }


    printf("audio stream configured, allocating neural network.\n");
    // setup neural network

    // settings
    NN_learning_settings* learning_settings = (NN_learning_settings*)malloc(sizeof(NN_learning_settings));
    NN_use_settings* use_settings = (NN_use_settings*)malloc(sizeof(NN_use_settings));
    learning_settings->learning_rate = LEARNING_RATE;
    use_settings->activation = SIGMOID;
    learning_settings->optimizer = GRADIENT_DESCENT;
    learning_settings->use_batching = true;
    use_settings->device_type = CPU;

    // init
    unsigned int neurons_per_layer[2] = {PARAMETERS,PARAMETERS};
    NN_network* net_real = NN_network_init(neurons_per_layer, 2);
    NN_network* net_imaginary = NN_network_init(neurons_per_layer, 2);
    NN_trainer* trainer_real = NN_trainer_init(net_real, learning_settings, use_settings, "cpu1");
    NN_trainer* trainer_imaginary = NN_trainer_init(net_imaginary, learning_settings, use_settings, "cpu1");
    
    if (NN_network_load_from_file(net_real, "real_latest.net")==-3) {
        // if no network to load from exists, just randomise
        NN_network_randomise(net_real, RANDOM_INIT_MIN, RANDOM_INIT_MAX, RANDOM_INIT_MIN, RANDOM_INIT_MAX);
    }

    if (NN_network_load_from_file(net_imaginary, "imaginary_latest.net")==-3) {
        NN_network_randomise(net_imaginary, RANDOM_INIT_MIN, RANDOM_INIT_MAX, RANDOM_INIT_MIN, RANDOM_INIT_MAX);
    }

    // malloc audio buffers
    float* clean = malloc(PARAMETERS * sizeof(float));
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

    // start training
    unsigned int epoch = 0;
    float loss = 0;
    for (unsigned int i = 0; i < 400000; i++) {
        loss = 0;
        drwav_seek_to_pcm_frame(&tts_speech, random_uint_range(0,tts_speech.totalPCMFrameCount-(BATCH_SIZE*PARAMETERS+1)));
        drwav_seek_to_pcm_frame(&background_noise, random_uint_range(0,background_noise.totalPCMFrameCount-(BATCH_SIZE*PARAMETERS+1)));
        drwav_seek_to_pcm_frame(&foreground_noise, random_uint_range(0,foreground_noise.totalPCMFrameCount-(BATCH_SIZE*PARAMETERS+1)));
        for (unsigned int batch = 0; batch < BATCH_SIZE; batch++) {

            drwav_read_pcm_frames_f32(&tts_speech, PARAMETERS, clean);
            drwav_read_pcm_frames_f32(&foreground_noise, PARAMETERS, foreground);
            drwav_read_pcm_frames_f32(&background_noise, PARAMETERS, background);

            // mix signals
            for (unsigned int i = 0; i < PARAMETERS; i++) {
                noisey[i] = (clean[i] + background[i] + foreground[i]) / 3;
            }

            


            //dft(noisey_ft, noisey, PARAMETERS);
            pthread_create(&dft_thread, NULL, dft_thread_func, (void*)&dft_thread_arg);
            dft(clean_ft, clean, PARAMETERS);
            pthread_join(dft_thread, NULL);
            
            pthread_create(&real_thread, NULL, trainer_thread_func, (void*)&trainer_thread_arg);
            //NN_trainer_accumulate(trainer_real, noisy_real,clean_real);
            NN_trainer_accumulate(trainer_imaginary, noisy_imag,clean_imag);
            pthread_join(real_thread, NULL);
            loss += NN_trainer_loss(trainer_real, clean_real) * 0.5;
            loss += NN_trainer_loss(trainer_imaginary, clean_imag) * 0.5;

            reconstruct(clean, (complex_array){.imaginary=trainer_imaginary->processor.network->out[trainer_imaginary->processor.network->layers-1],.real=trainer_real->processor.network->out[trainer_real->processor.network->layers-1],.size=PARAMETERS}, PARAMETERS);
            unsigned int left = PaUtil_GetRingBufferWriteAvailable(&ringBuffer);
            PaUtil_WriteRingBuffer(&ringBuffer, clean, min(PARAMETERS, left));
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

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    free(ringBufferData);
    drwav_uninit(&tts_speech);
    drwav_uninit(&background_noise);
    drwav_uninit(&foreground_noise);
    return 0;
}