clc;
clear;
close all;

%% 仿真参数设置
N_t = 2;         % 发射天线数
N_r = 2;         % 接收天线数
N_fft = 64;      % OFDM子载波数量
N_cp = 16;       % 循环前缀长度
N_frame = 10;    % 每个SNR点仿真的帧数
N_bits_per_frame = N_t * N_fft * 2; % 每帧的比特数 (QPSK为2 bit/symbol)
Total_bits = N_frame * N_bits_per_frame;

SNR_dB_range = 0:2:20; % 信噪比范围 (dB)
BER_ZF = zeros(size(SNR_dB_range));
BER_MMSE = zeros(size(SNR_dB_range));
BER_SISO = zeros(size(SNR_dB_range));

%% 仿真主循环
for i_snr = 1:length(SNR_dB_range)
    snr_db = SNR_dB_range(i_snr);
    snr_linear = 10^(snr_db / 10);
    total_errors_zf = 0;
    total_errors_mmse = 0;
    total_errors_siso = 0;
    disp(['Simulating SNR = ', num2str(snr_db), ' dB']);
    for i_frame = 1:N_frame
        %% 发射端
        % 产生原始比特流
        tx_bits = randi([0 1], 1, N_bits_per_frame);
        
        % QPSK调制 (00->-1-j, 01->-1+j, 10->1-j, 11->1+j)/sqrt(2)
        tx_symbols_serial = 1/sqrt(2) * ( (1-2*tx_bits(1:2:end)) + 1j * (1-2*tx_bits(2:2:end)) );
        
        % 串并转换，分配到各个天线和子载波
        tx_symbols_parallel = reshape(tx_symbols_serial, N_t, N_fft);
        
        % IFFT
        tx_signal_time = ifft(tx_symbols_parallel, N_fft, 2);
        
        % 添加循环前缀 (CP)
        tx_signal_with_cp = [tx_signal_time(:, N_fft-N_cp+1:N_fft), tx_signal_time];
        
        %% 信道
        % 生成瑞利衰落信道矩阵 H (N_r x N_t) for each subcarrier
        % 假设每个子载波的信道独立
        H = (randn(N_r, N_t, N_fft) + 1j * randn(N_r, N_t, N_fft)) / sqrt(2);
        
        %信号通过信道
        rx_signal_with_cp = zeros(N_r, N_fft + N_cp);
        for k = 1:N_fft
            %在时域，信道表现为多径。在频域，每个子载波对应一个H_k
            %这里为了简化，直接在频域进行操作，等效于时域卷积
            %实际中是时域信号与信道冲击响应卷积
        end
        % 简化模型：直接在频域处理
        % 添加高斯白噪声 (AWGN)
        % 噪声功率计算：信号功率归一化为1，Es = 1
        % SNR = Es / N0 => N0 = 1 / snr_linear
        noise_power = 1 / snr_linear;
        noise = sqrt(noise_power/2) * (randn(N_r, N_fft) + 1j * randn(N_r, N_fft));
        rx_symbols_freq = zeros(N_r, N_fft);
        for k = 1:N_fft
            rx_symbols_freq(:, k) = H(:, :, k) * tx_symbols_parallel(:, k) + noise(:, k);
        end
        
        %% 接收端
        rx_detected_symbols_zf = zeros(N_t, N_fft);
        rx_detected_symbols_mmse = zeros(N_t, N_fft);
        
        for k = 1:N_fft
            H_k = H(:, :, k);
            y_k = rx_symbols_freq(:, k);
            
            % ZF检测
            W_zf = inv(H_k' * H_k) * H_k';
            x_hat_zf = W_zf * y_k;
            rx_detected_symbols_zf(:, k) = x_hat_zf;
            
            % MMSE检测
            % W_mmse = inv(H_k' * H_k + (1/snr_linear) * eye(N_t)) * H_k';
            W_mmse = (H_k' * H_k + eye(N_t) / snr_linear) \ H_k'; % 更稳定的计算方式
            x_hat_mmse = W_mmse * y_k;
            rx_detected_symbols_mmse(:, k) = x_hat_mmse;
        end
        
        % QPSK解调
        % ZF
        rx_bits_zf_serial = demapper_qpsk(reshape(rx_detected_symbols_zf, 1, []));
        % MMSE
        rx_bits_mmse_serial = demapper_qpsk(reshape(rx_detected_symbols_mmse, 1, []));
        
        % 计算误码数
        total_errors_zf = total_errors_zf + sum(tx_bits ~= rx_bits_zf_serial);
        total_errors_mmse = total_errors_mmse + sum(tx_bits ~= rx_bits_mmse_serial);
    end
    
    % 计算BER
    BER_ZF(i_snr) = total_errors_zf / Total_bits;
    BER_MMSE(i_snr) = total_errors_mmse / Total_bits;
    
    % SISO理论BER (for QPSK in Rayleigh channel)
    % BER_SISO(i_snr) = 0.5 * (1 - sqrt(snr_linear / (2 + snr_linear)));
    % 用仿真得到SISO BER以保证公平性
    [~, BER_SISO(i_snr)] = berawgn(snr_db, 'psk', 4, 'nondiff'); % AWGN for reference
    % A more accurate Rayleigh SISO would be better, but this is a simple baseline
    % A simple simulation for SISO
    siso_tx_bits = randi([0 1], 1, N_bits_per_frame);
    siso_tx_sym = 1/sqrt(2) * ( (1-2*siso_tx_bits(1:2:end)) + 1j * (1-2*siso_tx_bits(2:2:end)) );
    siso_h = (randn + 1j*randn)/sqrt(2);
    siso_noise = sqrt(noise_power/2)*(randn(size(siso_tx_sym)) + 1j*randn(size(siso_tx_sym)));
    siso_rx_sym = siso_h .* siso_tx_sym + siso_noise;
    siso_rx_sym_eq = siso_rx_sym / siso_h; % perfect channel equalization
    siso_rx_bits = demapper_qpsk(siso_rx_sym_eq);
    BER_SISO(i_snr) = sum(siso_tx_bits ~= siso_rx_bits) / N_bits_per_frame;

end

%% 结果绘图
figure;
semilogy(SNR_dB_range, BER_SISO, '-o', 'LineWidth', 1.5, 'DisplayName', '1x1 SISO');
hold on;
semilogy(SNR_dB_range, BER_ZF, '-s', 'LineWidth', 1.5, 'DisplayName', '2x2 MIMO-ZF');
semilogy(SNR_dB_range, BER_MMSE, '-^', 'LineWidth', 1.5, 'DisplayName', '2x2 MIMO-MMSE');
grid on;
xlabel('信噪比 (SNR / dB)');
ylabel('误码率 (BER)');
title('2x2 MIMO-OFDM系统中ZF与MMSE算法的BER性能对比');
legend('Location', 'southwest');
axis([min(SNR_dB_range) max(SNR_dB_range) 10^-5 1]);

%% 辅助函数：QPSK解调器
function bits = demapper_qpsk(symbols)
    bits_real = real(symbols) > 0;
    bits_imag = imag(symbols) > 0;
    bits = zeros(1, 2 * length(symbols));
    bits(1:2:end) = ~bits_real; % 映射规则的反向
    bits(2:2:end) = ~bits_imag;
end