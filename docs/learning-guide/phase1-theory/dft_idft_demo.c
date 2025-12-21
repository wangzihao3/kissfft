#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#define PI 3.14159265358979323846

// 实现 DFT（练习4）
void my_dft(double *input, double complex *output, int N) {
    for (int k = 0; k < N; k++) {
        output[k] = 0;
        for (int n = 0; n < N; n++) {
            // 计算旋转因子: exp(-j*2*pi*k*n/N)
            double complex twiddle = cexp(-2.0 * PI * I * k * n / N);
            output[k] += input[n] * twiddle;
        }
    }
}

// 实现 IDFT（练习5）
void my_idft(double complex *input, double *output, int N) {
    for (int n = 0; n < N; n++) {
        double complex sum = 0;
        for (int k = 0; k < N; k++) {
            // 计算旋转因子: exp(j*2*pi*k*n/N)
            double complex twiddle = cexp(2.0 * PI * I * k * n / N);
            sum += input[k] * twiddle;
        }
        // 注意 1/N 的缩放因子
        output[n] = creal(sum) / N;
    }
}

// 打印复数数组
void print_complex_array(double complex *arr, int N, const char *label) {
    printf("%s:\n", label);
    for (int i = 0; i < N; i++) {
        printf("  X[%d] = %.2f %+.2fi\n", i, creal(arr[i]), cimag(arr[i]));
    }
    printf("\n");
}

// 打印实数数组
void print_real_array(double *arr, int N, const char *label) {
    printf("%s:\n", label);
    for (int i = 0; i < N; i++) {
        printf("  x[%d] = %.6f\n", i, arr[i]);
    }
    printf("\n");
}

// 测试函数
void test_dft_idft() {
    printf("=== 测试 1: 基本功能测试 ===\n");

    // 测试用例 1：简单脉冲信号
    double input1[] = {1, 2, 3, 4};
    int N1 = 4;
    double complex spectrum1[4];
    double output1[4];

    printf("测试信号 1: [1, 2, 3, 4]\n\n");

    // 计算 DFT
    my_dft(input1, spectrum1, N1);
    print_complex_array(spectrum1, N1, "DFT 结果");

    // 计算 IDFT
    my_idft(spectrum1, output1, N1);
    print_real_array(output1, N1, "IDFT 恢复的信号");

    // 计算误差
    printf("重构误差:\n");
    double max_error1 = 0;
    for (int i = 0; i < N1; i++) {
        double error = fabs(input1[i] - output1[i]);
        printf("  x[%d]: 原值=%.6f, 恢复值=%.6f, 误差=%.2e\n",
               i, input1[i], output1[i], error);
        if (error > max_error1) max_error1 = error;
    }
    printf("最大误差: %.2e\n\n", max_error1);

    printf("=== 测试 2: 验证理论计算 ===\n");

    // 测试用例 2：问题 2.1 的序列 [1, 1, 0, 0]
    double input2[] = {1, 1, 0, 0};
    int N2 = 4;
    double complex spectrum2[4];
    double output2[4];

    printf("测试信号 2: [1, 1, 0, 0]\n");
    printf("理论计算结果:\n");
    printf("  X[0] = 2 + 0i\n");
    printf("  X[1] = 1 - 1i\n");
    printf("  X[2] = 0 + 0i\n");
    printf("  X[3] = 1 + 1i\n\n");

    // 计算 DFT
    my_dft(input2, spectrum2, N2);
    print_complex_array(spectrum2, N2, "实际 DFT 结果");

    // 计算 IDFT
    my_idft(spectrum2, output2, N2);
    print_real_array(output2, N2, "IDFT 恢复的信号");

    // 验证重构
    int correct = 1;
    for (int i = 0; i < N2; i++) {
        if (fabs(input2[i] - output2[i]) > 1e-10) {
            correct = 0;
            break;
        }
    }
    printf("重构是否成功: %s\n\n", correct ? "是" : "否");

    printf("=== 测试 3: 余弦信号测试 ===\n");

    // 测试用例 3：余弦信号
    int N3 = 8;
    double input3[8];
    double complex spectrum3[8];
    double output3[8];

    // 生成一个周期的余弦信号
    for (int i = 0; i < N3; i++) {
        input3[i] = cos(2 * PI * i / N3);
    }

    print_real_array(input3, N3, "余弦信号 (一个周期)");

    // 计算 DFT
    my_dft(input3, spectrum3, N3);
    print_complex_array(spectrum3, N3, "DFT 结果");

    // 计算 IDFT
    my_idft(spectrum3, output3, N3);
    print_real_array(output3, N3, "IDFT 恢复的信号");

    // 计算误差
    printf("重构误差:\n");
    double max_error3 = 0;
    for (int i = 0; i < N3; i++) {
        double error = fabs(input3[i] - output3[i]);
        if (error > max_error3) max_error3 = error;
    }
    printf("最大误差: %.2e\n\n", max_error3);
}

// 演示 DFT/IDFT 的性质
void demonstrate_properties() {
    printf("=== 演示 DFT/IDFT 性质 ===\n\n");

    // 生成测试信号
    int N = 6;
    double signal[] = {1, 2, 3, 4, 5, 6};
    double complex spectrum[N];
    double recovered[N];

    print_real_array(signal, N, "原始信号");

    // DFT -> IDFT
    my_dft(signal, spectrum, N);
    print_complex_array(spectrum, N, "频谱");

    my_idft(spectrum, recovered, N);
    print_real_array(recovered, N, "恢复的信号");

    // 验证能量保持性（Parseval 定理）
    double time_energy = 0;
    for (int i = 0; i < N; i++) {
        time_energy += signal[i] * signal[i];
    }

    double freq_energy = 0;
    for (int i = 0; i < N; i++) {
        freq_energy += creal(spectrum[i] * conj(spectrum[i])) / N;
    }

    printf("能量验证:\n");
    printf("  时域能量: %.6f\n", time_energy);
    printf("  频域能量: %.6f\n", freq_energy);
    printf("  能量差异: %.2e\n", time_energy - freq_energy);
}

int main() {
    printf("DFT 和 IDFT 实现测试\n");
    printf("====================\n\n");

    // 运行测试
    test_dft_idft();

    printf("\n");
    demonstrate_properties();

    printf("\n实现说明:\n");
    printf("1. DFT 公式: X[k] = Σ(n=0 to N-1) x[n] * exp(-j*2π*k*n/N)\n");
    printf("2. IDFT 公式: x[n] = (1/N) * Σ(k=0 to N-1) X[k] * exp(j*2π*k*n/N)\n");
    printf("3. IDFT 包含 1/N 缩放因子以确保正确重构\n");
    printf("4. 使用 cexp() 函数计算复指数\n");
    printf("5. 对于实数输入，输出应为实数（可能有小的数值误差）\n");

    return 0;
}