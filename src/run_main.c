#include "RNG.h"
#include "raylib.h"
#include <iso646.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
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


#define RING_BUFFER_SIZE (1 << 17) // Must be power of 2
#define ZERO_OUTPUT_BUFFER 1

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

    size_t samplesNeeded = framesPerBuffer;// * audio->numChannels;
    size_t samplesAvailable = PaUtil_GetRingBufferReadAvailable(&ringBuffer);

    #if ZERO_OUTPUT_BUFFER
        // Zero out in case of underrun (means instead of hearing rancid sounds you just hear nothing)
        memset(outputBuffer, 0, samplesNeeded*sizeof(float));
    #endif

    size_t samplesToRead = (samplesAvailable < samplesNeeded) ? samplesAvailable : samplesNeeded;

    PaUtil_ReadRingBuffer(&ringBuffer, outputBuffer, samplesToRead);


    

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
        printf("Usage: %s <input.wav> <real.net> <imag.net>\n", argv[0]);
        return 1;
    }
    

    // init wav streams
    drwav input_wav;
    if (!drwav_init_file(&input_wav, argv[1], NULL)) {fprintf(stderr, "failed to open %s\n",argv[1]);return 1;}


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
    Pa_OpenStream(&stream, NULL, &outputParams, input_wav.sampleRate,
                  paFramesPerBufferUnspecified, paClipOff, paCallback, &audio);

    PaError err = Pa_StartStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "Pa_StartStream failed: %s\n", Pa_GetErrorText(err));
        return 1;
    }


    printf("audio stream configured, allocating neural networks.\n");
    // setup neural networks

    // settings
    NN_use_settings* use_settings = (NN_use_settings*)malloc(sizeof(NN_use_settings));
    use_settings->activation = TANH;
    use_settings->device_type = CPU;

    // init
    NN_network* net_real = NN_network_init_from_file(argv[2]);
    NN_network* net_imag = NN_network_init_from_file(argv[3]);
    NN_processor* processor_real = NN_processor_init(net_real, use_settings, "cpu1");
    NN_processor* processor_imag = NN_processor_init(net_imag, use_settings, "cpu1");

    if (net_real->neurons_per_layer[0] != net_imag->neurons_per_layer[0]) {
        fprintf(stderr, "parameter mismatch between networks\n");
    }

    size_t PARAMETERS = net_real->neurons_per_layer[0];

    // malloc audio buffers
    float* clean = malloc(PARAMETERS * sizeof(float));
    float* clean_imag = malloc(PARAMETERS * sizeof(float));
    float* clean_real = malloc(PARAMETERS * sizeof(float));
    complex_array clean_ft = {
        .imaginary = clean_imag,
        .real = clean_real,
        .size = PARAMETERS
    };

    // init display window
    InitWindow(PARAMETERS, 400, "fdart");


    unsigned int totalSamples = PARAMETERS * input_wav.channels;
    float* temp = malloc(totalSamples * sizeof(float));

    while (1) {
        unsigned int left = PaUtil_GetRingBufferWriteAvailable(&ringBuffer);
        if (left > PARAMETERS) {
            drwav_read_pcm_frames_f32(&input_wav, PARAMETERS, temp);
            // convert to mono
            for (unsigned int i = 0; i < PARAMETERS; i++) {
                float sum = 0.0f;
                for (unsigned int ch = 0; ch < input_wav.channels; ch++) {
                    sum += temp[i * input_wav.channels + ch];
                }
                clean[i] = sum / input_wav.channels;
            }

            // fft + normalise
            fft(clean_ft, clean, PARAMETERS);
            float original_rms_clean = rms_compute_complex_array(&clean_ft);
            rms_normalize_complex_array(&clean_ft, 0.25);
            // process
            NN_processor_process(processor_imag, clean_imag, clean_imag);
            NN_processor_process(processor_real, clean_real, clean_real);

            // denormalize + ifft + output
            rms_denormalize_complex_array(&clean_ft, original_rms_clean, 0.25f);
            ifft(clean_ft, clean, PARAMETERS);
            PaUtil_WriteRingBuffer(&ringBuffer, clean,
                    min(PARAMETERS, left));
        } else {
            Pa_Sleep(20);
        }
    }
    


    
    


    // clean up
    free(temp);

    free(clean);

    free(clean_imag);
    free(clean_real);

    NN_processor_free(processor_imag);
    NN_processor_free(processor_real);

    NN_network_free(net_imag);
    NN_network_free(net_real);

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    free(ringBufferData);
    drwav_uninit(&input_wav);
    CloseWindow();
    return 0;
}