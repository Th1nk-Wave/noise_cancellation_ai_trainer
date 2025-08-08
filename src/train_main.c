#define DR_WAV_IMPLEMENTATION

#include <stdio.h>
#include <portaudio.h>
#include <pa_ringbuffer.h>
#include <dr_wav.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

#define RING_BUFFER_SIZE (1 << 16) // Must be power of 2
#define ZERO_OUTPUT_BUFFER 0

static PaUtilRingBuffer ringBuffer;
static void* ringBufferData;

typedef struct {
    int numChannels;
    PaSampleFormat format;
} AudioState;

static int paCallback(const void *inputBuffer, void *outputBuffer,
                      unsigned long framesPerBuffer,
                      const PaStreamCallbackTimeInfo* timeInfo,
                      PaStreamCallbackFlags statusFlags,
                      void *userData)
{
    AudioState* audio = (AudioState*)userData;

    size_t bytesPerFrame = Pa_GetSampleSize(audio->format) * audio->numChannels;
    size_t bytesToWrite = framesPerBuffer * bytesPerFrame;

    #if ZERO_OUTPUT_BUFFER
    // Zero out in case of underrun (means instead of hearing rancid sounds you just hear nothing)
    memset(outputBuffer, 0, bytesToWrite);
    #endif

    size_t bytesAvailable = PaUtil_GetRingBufferReadAvailable(&ringBuffer);
    size_t bytesToRead = (bytesAvailable < bytesToWrite) ? bytesAvailable : bytesToWrite;

    PaUtil_ReadRingBuffer(&ringBuffer, outputBuffer, bytesToRead);

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
    if (argc < 2) {
        printf("Usage: %s <file.wav>\n", argv[0]);
        return 1;
    }
    

    // init wav reader
    drwav wav;
    if (!drwav_init_file(&wav, argv[1], NULL)) {
        fprintf(stderr, "failed to open %s\n",argv[1]);
        return 1;
    }

    printf("sample rate: %i\n"
            "bits per sample %i\n"
            "channels: %i\n"
            "frames: %llu\n"
        
        ,wav.sampleRate,wav.bitsPerSample,wav.channels,wav.totalPCMFrameCount);


    // init portAudio
    printf("initialising audio interface...\n");
    AudioState audio = {
        .numChannels = wav.channels,
        .format = paInt16,
    };

    ringBufferData = malloc(RING_BUFFER_SIZE);
    PaUtil_InitializeRingBuffer(&ringBuffer, 1, RING_BUFFER_SIZE, ringBufferData);
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
    if (deviceInfo->maxOutputChannels < wav.channels) {
        fprintf(stderr, "No suitable output device found\nenter device number you want to use: ");
        scanf("%i",&selectedDevice);
        deviceInfo = Pa_GetDeviceInfo(selectedDevice);
        host = Pa_GetHostApiInfo(deviceInfo->hostApi);
    }
    printf("Using device [%d]: %s (%s)\n", selectedDevice, deviceInfo->name, host->name);

    // configure device stream
    PaStreamParameters outputParams = {
        .device = selectedDevice,
        .channelCount = wav.channels,
        .sampleFormat = paInt16,
        .suggestedLatency = deviceInfo->defaultLowOutputLatency,
        .hostApiSpecificStreamInfo = NULL
    };

    PaStream* stream;
    Pa_OpenStream(&stream, NULL, &outputParams, wav.sampleRate,
                  paFramesPerBufferUnspecified, paClipOff, paCallback, &audio);

    PaError err = Pa_StartStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "Pa_StartStream failed: %s\n", Pa_GetErrorText(err));
        return 1;
    }

    pthread_t reader;
    pthread_create(&reader, NULL, readerThreadFunc, (void*)&wav);

    while (Pa_IsStreamActive(stream)) {
        Pa_Sleep(100);
    }


    pthread_join(reader, NULL);
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    free(ringBufferData);
    drwav_uninit(&wav);
    return 0;

}