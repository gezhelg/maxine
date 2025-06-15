%% 主仿真框架
clear; clc;
% 系统参数
Nt = 2; Nr = 2;           % 天线数
Nfft = 64; CP_len = 16;   % OFDM参数
SNR_dB = 0:5:30;          % 信噪比范围
mod_order = 4;            % QPSK调制
L = 3;                    % 多径数
path_delays = [0, 1, 3];  % 时延(samples)
path_gains = [1, 0.5, 0.25]; % 路径增益(线性)
max_err = 1000;           % 最大错误比特数

% 结果存储
ber_zf = zeros(size(SNR_dB));
ber_mmse = zeros(size(SNR_dB));

for snr_idx = 1:length(SNR_dB)
    err_zf = 0; err_mmse = 0; total_bits = 0;
    snr_linear = 10^(SNR_dB(snr_idx)/10);
    noise_var = 1/snr_linear;  % 发送符号功率归一化为1
    
    while (err_zf < max_err) || (err_mmse < max_err)
        % 生成随机数据
        data_bits = randi([0 1], Nt*Nfft*log2(mod_order), 1);
        
        % QPSK调制
        tx_sym = pskmod(data_bits, mod_order, InputType='bit');
        tx_sym = reshape(tx_sym, Nt, Nfft); % Nt x Nfft
        
        % 生成频率选择性MIMO信道
        H_freq = zeros(Nr, Nt, Nfft);
        for rx = 1:Nr
            for tx = 1:Nt
                % 生成时域冲激响应
                h_t = (randn(1,L) + 1i*randn(1,L)) .* sqrt(path_gains/2);
                % 转换到频域
                H_freq(rx,tx,:) = fft(h_t, Nfft);
            end
        end
        
        % 频域传输
        rx_sym = zeros(Nr, Nfft);
        for k = 1:Nfft
            Hk = H_freq(:,:,k);
            rx_sym(:,k) = Hk * tx_sym(:,k);
        end
        
        % 添加高斯噪声
        noise = sqrt(noise_var/2)*(randn(Nr,Nfft) + 1i*randn(Nr,Nfft));
        rx_sym = rx_sym + noise;
        
        % MIMO检测
        dec_bits_zf = [];
        dec_bits_mmse = [];
        for k = 1:Nfft
            Hk = squeeze(H_freq(:,:,k));
            % ZF检测
            W_zf = pinv(Hk);
            dec_zf = W_zf * rx_sym(:,k);
            % MMSE检测
            W_mmse = (Hk'*Hk + noise_var*eye(Nt)) \ Hk';
            dec_mmse = W_mmse * rx_sym(:,k);
            
            % 解调
            bits_zf = pskdemod(dec_zf, mod_order, OutputType='bit');
            bits_mmse = pskdemod(dec_mmse, mod_order, OutputType='bit');
            
            dec_bits_zf = [dec_bits_zf; bits_zf];
            dec_bits_mmse = [dec_bits_mmse; bits_mmse];
        end
        
        % 误码统计
        err_zf = err_zf + sum(data_bits ~= dec_bits_zf);
        err_mmse = err_mmse + sum(data_bits ~= dec_bits_mmse);
        total_bits = total_bits + length(data_bits);
    end
    
    ber_zf(snr_idx) = err_zf / total_bits;
    ber_mmse(snr_idx) = err_mmse / total_bits;
end

%% 绘图
figure;
semilogy(SNR_dB, ber_zf, 'r-o', 'LineWidth', 2); hold on;
semilogy(SNR_dB, ber_mmse, 'b-s', 'LineWidth', 2);
grid on; 
xlabel('SNR (dB)'); 
ylabel('BER');
legend('ZF', 'MMSE'); 
title('MIMO检测算法性能对比');
set(gca, 'YScale', 'log');

%% SVD预编码与功率分配
function [capacity] = svd_waterfilling(H, SNR_dB)
    % 信道分解
    [~, S, V] = svd(H);
    lambda = diag(S).^2;  % 信道增益
    
    % 注水功率分配
    noise_var = 10^(-SNR_dB/10);
    N = min(size(H));
    mu = waterfill_mu(lambda, noise_var, N);
    
    % 计算容量
    P_k = max(0, mu - noise_var./lambda);
    capacity = sum(log2(1 + P_k.*lambda/noise_var));
end

function mu = waterfill_mu(lambda, N0, N)
    % 注水算法求解mu
    sorted_lambda = sort(lambda, 'descend');
    cumulative = 0;
    for k = 1:N
        cumulative = cumulative + N0/sorted_lambda(k);
        mu = (1 + cumulative)/k;
        if k < N
            if mu < N0/sorted_lambda(k+1)
                break;
            end
        end
    end
end

%% Alamouti编码
function [ber] = alamouti_sim(SNR_dB, N_symbols)
    % 参数设置
    mod_order = 4; % QPSK
    noise_var = 10^(-SNR_dB/10);
    total_bits = 0;
    errors = 0;
    
    for sym = 1:N_symbols
        % 生成发送符号
        s1 = pskmod(randi([0 3]), mod_order);
        s2 = pskmod(randi([0 3]), mod_order);
        
        % Alamouti编码
        C = [s1, -conj(s2); 
             s2, conj(s1)];
        
        % 信道响应
        h1 = (randn + 1i*randn)/sqrt(2);
        h2 = (randn + 1i*randn)/sqrt(2);
        H = [h1, h2];
        
        % 接收信号
        y = H * C + sqrt(noise_var/2)*(randn(2,1) + 1i*randn(2,1));
        
        % 最大似然检测
        s_hat = zeros(2,1);
        s_hat(1) = conj(h1)*y(1) + h2*conj(y(2));
        s_hat(2) = conj(h2)*y(1) - h1*conj(y(2));
        norm_factor = abs(h1)^2 + abs(h2)^2;
        s_hat = s_hat / norm_factor;
        
        % 解调
        bits_tx = [pskdemod(s1, mod_order, OutputType='bit'); 
                   pskdemod(s2, mod_order, OutputType='bit')];
        bits_rx = [pskdemod(s_hat(1), mod_order, OutputType='bit');
                   pskdemod(s_hat(2), mod_order, OutputType='bit')];
        
        errors = errors + sum(bits_tx ~= bits_rx);
        total_bits = total_bits + length(bits_tx);
    end
    
    ber = errors / total_bits;
end