#ifndef __FILE_STREAM_H_
#define __FILE_STREAM_H_

#include <dr_wav.h>
#include <pa_ringbuffer.h>
#include <pthread.h>

enum file_stream_command_type {
    PAUSE,
    PLAY,
    SEEK,
};

typedef struct {
    enum file_stream_command_type type;
    void* args;
} file_stream_command;

typedef struct {
    drwav* input;
    PaUtilRingBuffer* output; 
    file_stream_command* command_buffer;
} file_stream_thread_args;

typedef struct {
    file_stream_thread_args args;

} file_stream;

file_stream init_file_stream(drwav* input, PaUtilRingBuffer* output);

void pause_file_stream(file_stream* fs);
void play_file_stream(file_stream* fs);
void seek_file_stream(file_stream* fs, unsigned int position);

#endif