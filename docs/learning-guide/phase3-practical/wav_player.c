/*
 * WAV 文件播放和处理工具
 * 展示如何将 KISS FFT 应用于实际音频处理
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kiss_fft.h"
#include "kiss_fftr.h"

#pragma pack(push, 1)
typedef struct {
    char riff[4];          // "RIFF"
    uint32_t file_size;    // 文件大小 - 8
    char wave[4];          // "WAVE"
    char fmt[4];           // "fmt "
    uint32_t fmt_size;     // fmt 块大小 (16)
    uint16_t audio_format; // 音频格式 (1 = PCM)
    uint16_t num_channels; // 声道数
    uint32_t sample_rate;  // 采样率
    uint32_t byte_rate;    // 字节率
    uint16_t block_align;  // 块对齐
    uint16_t bits_per_sample; // 位深度
    char data[4];          // "data"
    uint32_t data_size;    // 数据大小
} WAVHeader;
#pragma pack(pop)

typedef struct {
    WAVHeader header;
    float *samples;        // 浮点样本数据
    int num_samples;       // 样本数量
    int duration_ms;       // 时长（毫秒）
} AudioFile;

// 读取 WAV 文件
AudioFile* wav_load(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }

    AudioFile *audio = malloc(sizeof(AudioFile));
    fread(&audio->header, sizeof(WAVHeader), 1, file);

    // 验证 WAV 格式
    if (strncmp(audio->header.riff, "RIFF", 4) != 0 ||
        strncmp(audio->header.wave, "WAVE", 4) != 0) {
        printf("Error: Not a valid WAV file\n");
        free(audio);
        fclose(file);
        return NULL;
    }

    // 计算样本数量
    int bytes_per_sample = audio->header.bits_per_sample / 8;
    audio->num_samples = audio->header.data_size / (bytes_per_sample * audio->header.num_channels);
    audio->duration_ms = (audio->num_samples * 1000) / audio->header.sample_rate;

    // 分配样本缓冲区
    audio->samples = malloc(sizeof(float) * audio->num_samples);

    // 读取样本数据
    if (audio->header.bits_per_sample == 16) {
        int16_t *buffer = malloc(audio->header.data_size);
        fread(buffer, 1, audio->header.data_size, file);

        // 转换为浮点数 (-1.0 到 1.0)
        for (int i = 0; i < audio->num_samples; i++) {
            audio->samples[i] = (float)buffer[i] / 32768.0f;
        }

        free(buffer);
    } else if (audio->header.bits_per_sample == 32) {
        int32_t *buffer = malloc(audio->header.data_size);
        fread(buffer, 1, audio->header.data_size, file);

        for (int i = 0; i < audio->num_samples; i++) {
            audio->samples[i] = (float)buffer[i] / 2147483648.0f;
        }

        free(buffer);
    } else {
        printf("Error: Unsupported bit depth: %d\n", audio->header.bits_per_sample);
        free(audio->samples);
        free(audio);
        fclose(file);
        return NULL;
    }

    fclose(file);

    // 如果是立体声，转换为单声道（平均值）
    if (audio->header.num_channels == 2) {
        float *mono = malloc(sizeof(float) * audio->num_samples / 2);
        for (int i = 0; i < audio->num_samples / 2; i++) {
            mono[i] = (audio->samples[2*i] + audio->samples[2*i+1]) * 0.5f;
        }
        free(audio->samples);
        audio->samples = mono;
        audio->num_samples /= 2;
        audio->header.num_channels = 1;
    }

    return audio;
}

// 打印音频信息
void wav_print_info(AudioFile *audio) {
    printf("=== Audio File Information ===\n");
    printf("Format: PCM %d-bit\n", audio->header.bits_per_sample);
    printf("Channels: %d\n", audio->header.num_channels);
    printf("Sample Rate: %d Hz\n", audio->header.sample_rate);
    printf("Duration: %d ms (%.2f seconds)\n", audio->duration_ms, audio->duration_ms / 1000.0f);
    printf("Samples: %d\n", audio->num_samples);
    printf("===============================\n\n");
}

// 保存为新的 WAV 文件
void wav_save(const char *filename, AudioFile *audio) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Cannot create file %s\n", filename);
        return;
    }

    // 更新头部信息
    audio->header.data_size = audio->num_samples * sizeof(int16_t);
    audio->header.file_size = sizeof(WAVHeader) + audio->header.data_size - 8;

    // 写入头部
    fwrite(&audio->header, sizeof(WAVHeader), 1, file);

    // 转换并写入样本数据
    int16_t *buffer = malloc(audio->header.data_size);
    for (int i = 0; i < audio->num_samples; i++) {
        float sample = fmaxf(-1.0f, fminf(1.0f, audio->samples[i]));
        buffer[i] = (int16_t)(sample * 32767.0f);
    }
    fwrite(buffer, 1, audio->header.data_size, file);
    free(buffer);

    fclose(file);
    printf("Saved: %s\n", filename);
}

// 应用低通滤波器
void apply_lowpass_filter(AudioFile *audio, float cutoff_freq) {
    int fft_size = 4096;
    kiss_fftr_cfg fft_cfg = kiss_fftr_alloc(fft_size, 0, NULL, NULL);
    kiss_fftr_cfg ifft_cfg = kiss_fftr_alloc(fft_size, 1, NULL, NULL);

    float *fft_in = malloc(sizeof(float) * fft_size);
    kiss_fft_cpx *fft_out = malloc(sizeof(kiss_fft_cpx) * (fft_size/2 + 1));
    float *ifft_out = malloc(sizeof(float) * fft_size);

    // 计算截止频率对应的 FFT bin
    int cutoff_bin = (int)(cutoff_freq * fft_size / audio->header.sample_rate);
    cutoff_bin = fmin(cutoff_bin, fft_size/2);

    printf("Applying lowpass filter: cutoff = %.0f Hz (bin %d)\n", cutoff_freq, cutoff_bin);

    // 分块处理
    int hop_size = fft_size / 2;
    int num_blocks = (audio->num_samples - fft_size) / hop_size + 1;

    float *window = malloc(sizeof(float) * fft_size);
    for (int i = 0; i < fft_size; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (fft_size - 1))); // 汉宁窗
    }

    for (int block = 0; block < num_blocks; block++) {
        int start = block * hop_size;

        // 复制数据并应用窗函数
        for (int i = 0; i < fft_size; i++) {
            fft_in[i] = (start + i < audio->num_samples) ?
                       audio->samples[start + i] * window[i] : 0.0f;
        }

        // FFT
        kiss_fftr(fft_cfg, fft_in, fft_out);

        // 应用低通滤波
        for (int k = 0; k <= cutoff_bin; k++) {
            // 保持低频不变
        }
        for (int k = cutoff_bin + 1; k < fft_size/2 + 1; k++) {
            // 衰减高频
            fft_out[k].r *= 0.01f;
            fft_out[k].i *= 0.01f;
        }

        // IFFT
        kiss_fftri(ifft_cfg, fft_out, ifft_out);

        // 重叠相加
        for (int i = 0; i < fft_size; i++) {
            if (start + i < audio->num_samples) {
                audio->samples[start + i] = ifft_out[i] / fft_size * window[i];
            }
        }
    }

    free(window);
    free(fft_in);
    free(fft_out);
    free(ifft_out);
    kiss_fftr_free(fft_cfg);
    kiss_fftr_free(ifft_cfg);
}

// 生成频谱图
void generate_spectrogram(AudioFile *audio, const char *output_filename) {
    const int fft_size = 1024;
    const int hop_size = 512;
    const int num_bins = fft_size / 2 + 1;

    int num_frames = (audio->num_samples - fft_size) / hop_size + 1;

    printf("Generating spectrogram: %d frames x %d bins\n", num_frames, num_bins);

    kiss_fftr_cfg fft_cfg = kiss_fftr_alloc(fft_size, 0, NULL, NULL);

    float *fft_in = malloc(sizeof(float) * fft_size);
    kiss_fft_cpx *fft_out = malloc(sizeof(kiss_fft_cpx) * num_bins);
    float *spectrogram = malloc(sizeof(float) * num_frames * num_bins);

    // 窗函数
    float *window = malloc(sizeof(float) * fft_size);
    for (int i = 0; i < fft_size; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (fft_size - 1)));
    }

    // 计算每一帧的频谱
    for (int frame = 0; frame < num_frames; frame++) {
        int start = frame * hop_size;

        // 准备输入数据
        for (int i = 0; i < fft_size; i++) {
            fft_in[i] = (start + i < audio->num_samples) ?
                       audio->samples[start + i] * window[i] : 0.0f;
        }

        // FFT
        kiss_fftr(fft_cfg, fft_in, fft_out);

        // 计算对数幅值
        for (int k = 0; k < num_bins; k++) {
            float magnitude = sqrtf(fft_out[k].r * fft_out[k].r +
                                  fft_out[k].i * fft_out[k].i);
            spectrogram[frame * num_bins + k] = 20.0f * log10f(magnitude + 1e-10f);
        }
    }

    // 保存为简单的文本格式（可以用 Python 或 Excel 绘图）
    FILE *file = fopen(output_filename, "w");
    if (file) {
        fprintf(file, "# Spectrogram data\n");
        fprintf(file, "# Frame, Freq(Hz), Magnitude(dB)\n");

        for (int frame = 0; frame < num_frames; frame += 10) { // 每10帧取一个
            for (int k = 0; k < num_bins; k += 4) { // 每4个bin取一个
                float freq = (float)k * audio->header.sample_rate / fft_size;
                float mag = spectrogram[frame * num_bins + k];
                fprintf(file, "%d, %.1f, %.2f\n", frame, freq, mag);
            }
            fprintf(file, "\n"); // 空行分隔帧
        }

        fclose(file);
        printf("Spectrogram saved to: %s\n", output_filename);
    }

    free(window);
    free(fft_in);
    free(fft_out);
    free(spectrogram);
    kiss_fftr_free(fft_cfg);
}

// 音频增强（简单的动态范围压缩）
void audio_enhance(AudioFile *audio) {
    // 计算音频的 RMS
    float sum_squares = 0.0f;
    for (int i = 0; i < audio->num_samples; i++) {
        sum_squares += audio->samples[i] * audio->samples[i];
    }
    float rms = sqrtf(sum_squares / audio->num_samples);

    printf("Audio RMS: %.6f\n", rms);

    // 如果 RMS 太小，进行增益
    float target_rms = 0.1f;
    if (rms < target_rms) {
        float gain = target_rms / (rms + 1e-10f);
        printf("Applying gain: %.2f\n", gain);

        for (int i = 0; i < audio->num_samples; i++) {
            audio->samples[i] *= gain;
        }
    }

    // 软限幅
    float threshold = 0.95f;
    for (int i = 0; i < audio->num_samples; i++) {
        if (fabsf(audio->samples[i]) > threshold) {
            audio->samples[i] = copysignf(threshold, audio->samples[i]);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input.wav> [output.wav]\n", argv[0]);
        printf("Examples:\n");
        printf("  %s input.wav                    # Print info and generate spectrogram\n", argv[0]);
        printf("  %s input.wav output.wav         # Process and save\n", argv[0]);
        printf("\nProcessing options (when output specified):\n");
        printf("  -filter <freq>    Apply lowpass filter (Hz)\n");
        printf("  -enhance          Audio enhancement\n");
        return 1;
    }

    // 加载音频文件
    AudioFile *audio = wav_load(argv[1]);
    if (!audio) {
        return 1;
    }

    wav_print_info(audio);

    // 生成频谱图
    char spectrogram_name[256];
    snprintf(spectrogram_name, sizeof(spectrogram_name),
             "%s_spectrogram.txt", argv[1]);
    generate_spectrogram(audio, spectrogram_name);

    // 如果有输出文件，进行音频处理
    if (argc > 2) {
        // 解析处理选项
        int apply_filter = 0;
        int enhance = 0;
        float filter_freq = 0.0f;

        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "-filter") == 0 && i + 1 < argc) {
                apply_filter = 1;
                filter_freq = atof(argv[++i]);
            } else if (strcmp(argv[i], "-enhance") == 0) {
                enhance = 1;
            }
        }

        // 应用处理
        if (enhance) {
            printf("\nApplying audio enhancement...\n");
            audio_enhance(audio);
        }

        if (apply_filter) {
            printf("\nApplying filter...\n");
            apply_lowpass_filter(audio, filter_freq);
        }

        // 保存处理后的音频
        wav_save(argv[2], audio);
    }

    // 清理
    free(audio->samples);
    free(audio);

    return 0;
}