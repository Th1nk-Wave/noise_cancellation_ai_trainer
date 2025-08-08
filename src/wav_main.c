#include "util.h"
// probably some of the only well documented code i've ever written


/*
    writes the wav header to a given file.
*/
void write_wav_header(FILE *f, uint32_t data_size, uint16_t channels, uint32_t sample_rate, uint16_t bits_per_sample) {
    uint32_t byte_rate = sample_rate * channels * bits_per_sample / 8;
    uint16_t block_align = channels * bits_per_sample / 8;
    uint32_t chunk_size = 36 + data_size;

    // RIFF header
    fwrite("RIFF", 1, 4, f);                // RIFF signiture
    fwrite(&chunk_size, 4, 1, f);           // chunk size
    fwrite("WAVE", 1, 4, f);                // format specifier

    // fmt subchunk
    fwrite("fmt ", 1, 4, f);                // fmt subchunk
    uint32_t fmt_subchunk_size = 16;
    uint16_t audio_format = 1; // PCM
    fwrite(&fmt_subchunk_size, 4, 1, f);    // fmt chunk size
    fwrite(&audio_format, 2, 1, f);         // 1=pcm format type
    fwrite(&channels, 2, 1, f);             // number of channels (1=mono, 2=stereo, etc..)
    fwrite(&sample_rate, 4, 1, f);          // sample rate (8000Hz, 44100Hz etc..)
    fwrite(&byte_rate, 4, 1, f);            // Avg bytes per second (sample_rate * num_channels * bits_per_sample/8)
    fwrite(&block_align, 2, 1, f);          // block align (offset between each block of audio data)
    fwrite(&bits_per_sample, 2, 1, f);      // bits per sample (sample ranges between 0-1, higher bits per sample allows for more accuracy to represent that fraction)

    // data subchunk
    fwrite("data", 1, 4, f);                // data subchunk
    fwrite(&data_size, 4, 1, f);            // total size of the audio data in bytes
                                                           // what comes after this would be the pcm data
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input.sw output.wav\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];

    FILE *in = fopen(input_path, "rb");
    if (!in) {
        perror("fopen input");
        return 1;
    }

    fseek(in, 0, SEEK_END);
    uint32_t data_size = ftell(in);
    rewind(in);

    FILE *out = fopen(output_path, "wb");
    if (!out) {
        perror("fopen output");
        fclose(in);
        return 1;
    }

    const uint16_t channels = 1;
    const uint32_t sample_rate = 44100;
    const uint16_t bits_per_sample = 16;

    write_wav_header(out, data_size, channels, sample_rate, bits_per_sample);

    // copy data
    uint8_t buffer[4096];
    size_t read;
    while ((read = fread(buffer, 1, sizeof(buffer), in)) > 0) {
        fwrite(buffer, 1, read, out);
    }

    fclose(in);
    fclose(out);
    return 0;
}