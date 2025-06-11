#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pyvisa
import numpy as np
import pandas as pd
import time
import datetime
import threading
import queue
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import shutil
import jinja2
import concurrent.futures
from typing import Dict, List, Optional, Tuple
from doherty_config import DohertlyConfig
from doherty_manager import DohertlyManager, PowerMeasurement
from visualization import PAVisualization  # 导入新的可视化模块


class InstrumentManager(DohertlyManager):
    """测试仪器管理类，继承自DohertlyManager"""
    def __init__(self, config_file: str):
        doherty_config = DohertlyConfig(config_file)
        rm = pyvisa.ResourceManager()
        super().__init__(doherty_config, rm)
        
    def setup_test_instruments(self, freq_hz: float) -> bool:
        """设置所有测试仪器"""
        try:
            # 连接所有仪器
            self.connect_instruments()
            
            # 设置功率计
            self.setup_power_meter(freq_hz)
            
            # 设置驱动功放偏置
            if not self.setup_driver_bias():
                return False
                
            # 设置被测功放偏置
            if not self.setup_doherty_bias():
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"设置测试仪器失败: {e}")
            self.disconnect_all()
            raise
            
    def cleanup(self):
        """清理所有仪器连接"""
        self.disconnect_all()
        if hasattr(self, 'rm'):
            self.rm.close()

class DataManager:
    """数据管理类"""
    def __init__(self, config: dict):
        self.config = config
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(config['output_config']['base_dir']) / self.timestamp
        self.backup_dir = Path(config['output_config']['backup_dir'])
        self.setup_directories()
        
    def setup_directories(self):
        """创建必要的目录"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / 'charts').mkdir(exist_ok=True)
        
    def save_test_data(self, data: pd.DataFrame):
        """保存测试数据"""
        data.to_csv(self.base_dir / 'test_results.csv', index=False)
        self._backup_data('test_results.csv')
        
    def save_summary(self, summary: dict):
        """保存测试总结"""
        with open(self.base_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
            
    def _backup_data(self, filename: str):
        """备份数据文件"""
        src = self.base_dir / filename
        dst = self.backup_dir / f"{self.timestamp}_{filename}"
        shutil.copy2(src, dst)

class TestExecutor:
    """测试执行器类"""
    def __init__(self, config_file: str):
        """
        初始化测试执行器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.instrument_manager = InstrumentManager(config_file)
        self.data_manager = DataManager(self.instrument_manager.test_config.config)
        self.visualizer = Visualizer(self.data_manager)
        self.test_data = []
        self.recovery_file = Path('recovery.json')
        
    def run_test(self):
        """运行测试"""
        try:
            if self.recovery_file.exists():
                if self.load_recovery_point():
                    logging.info("从恢复点继续测试")
                else:
                    logging.warning("恢复点无效，从头开始测试")
                    self._setup_test()
            else:
                self._setup_test()
                
            if self.instrument_manager.test_config.config['test_config'].get('parallel_test', False):
                self._run_parallel_test()
            else:
                self._run_sequential_test()
                
        except Exception as e:
            logging.error(f"测试执行失败: {e}")
            raise
            
        finally:
            self._cleanup()
            
    def _setup_test(self):
        """测试准备"""
        self.test_data = []
        
    def _run_sequential_test(self):
        """顺序执行测试"""
        frequencies = self.instrument_manager.test_config.config['test_config']['frequencies']
        total_freqs = len(frequencies)
        
        for i, freq in enumerate(frequencies, 1):
            logging.info(f"测试频率 {freq/1e6:.1f}MHz ({i}/{total_freqs})")
            
            try:
                results = self._test_single_frequency(freq)
                self.test_data.extend(results)
                
                # 保存当前进度
                self._save_recovery_point(freq, results)
                
                # 保存阶段性结果
                if i % 5 == 0 or i == total_freqs:
                    df = pd.DataFrame(self.test_data)
                    self.data_manager.save_test_data(df)
                    
            except Exception as e:
                logging.error(f"频率{freq/1e6:.1f}MHz测试失败: {e}")
                raise
                
        # 生成测试报告
        df = pd.DataFrame(self.test_data)
        self.data_manager.save_test_data(df)
        summary = calculate_summary(df)
        self.data_manager.save_summary(summary)
        self.visualizer.generate_all_charts(df)
        self.visualizer.generate_html_report(df, summary)
        
    def _test_single_frequency(self, freq: float) -> List[dict]:
        """测试单个频率点"""
        results = []
        power_range = self.instrument_manager.test_config.config['test_config']['power_range']
        
        try:
            # 设置仪器
            if not self.instrument_manager.setup_test_instruments(freq):
                raise RuntimeError("仪器设置失败")
            
            # 设置信号源
            signal_gen = self.instrument_manager.instruments['signal_gen']
            signal_gen.write('*RST')
            signal_gen.write(f'FREQ {freq}Hz')
            signal_gen.write('POW -20dBm')  # 初始功率设置为最小值
            signal_gen.write('OUTP:STAT ON')
            time.sleep(0.5)
            
            # 设置频谱分析仪
            spectrum = self.instrument_manager.instruments['spectrum_analyzer']
            spectrum.write(f'FREQ:CENT {freq}Hz')
            spectrum.write('FREQ:SPAN 400MHz')
            spectrum.write('BAND:RES 300kHz')
            spectrum.write('BAND:VID 1MHz')
            spectrum.write('CALC:MARK:MAX')
            time.sleep(0.5)
            
            for power in np.arange(power_range['start'], 
                                 power_range['stop'] + power_range['step'], 
                                 power_range['step']):
                                 
                # 设置输入功率
                signal_gen.write(f'POW {power}dBm')
                time.sleep(0.2)
                
                # 测量输入功率
                input_power = None
                if self.instrument_manager.test_config.config['test_config']['driver_mode']:
                    input_power = self.instrument_manager.read_input_power()
                    if input_power is None:
                        logging.warning(f"输入功率{power}dBm读取失败")
                        continue
                else:
                    input_power = power
                
                # 测量输出功率
                output_power = float(spectrum.query('CALC:MARK:Y?'))
                
                # 测量各级功放功耗
                power_measurements = self.instrument_manager.measure_total_power()
                total_dc_power = sum(m.power for m in power_measurements.values())
                
                # 计算效率
                rf_output_power = 10 ** ((output_power - 30) / 10)  # dBm转W
                efficiency = rf_output_power / total_dc_power * 100 if total_dc_power > 0 else 0
                
                # 记录结果
                result = {
                    'frequency': freq,
                    'input_power': input_power,
                    'output_power': output_power,
                    'efficiency': efficiency,
                    'dc_power': total_dc_power,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                # 添加各级功放的详细数据
                for name, measurement in power_measurements.items():
                    result.update({
                        f'{name}_voltage': measurement.voltage,
                        f'{name}_current': measurement.current,
                        f'{name}_power': measurement.power
                    })
                    
                results.append(result)
                logging.info(f"Pin={input_power:.1f}dBm, "
                           f"Pout={output_power:.1f}dBm, "
                           f"Eff={efficiency:.1f}%")
                           
            return results
            
        except Exception as e:
            logging.error(f"频率{freq/1e6:.1f}MHz测试失败: {e}")
            raise
            
    def _cleanup(self):
        """清理资源"""
        self.instrument_manager.cleanup()
        if self.recovery_file.exists():
            self.recovery_file.unlink()

class Visualizer:
    """数据可视化类"""
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        
    def setup_figure_style(self):
        """设置图表样式"""
        plt.style.use('seaborn-darkgrid')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10

    def save_figure(self, fig: plt.Figure, name: str):
        """保存图表"""
        charts_dir = self.data_manager.base_dir / 'charts'
        for fmt in self.data_manager.config['output_config']['chart_formats']:
            if fmt == 'png':
                fig.savefig(charts_dir / f'{name}.png', dpi=300, bbox_inches='tight')
            elif fmt == 'pdf':
                fig.savefig(charts_dir / f'{name}.pdf', bbox_inches='tight')

    def generate_all_charts(self, data: pd.DataFrame):
        """生成所有图表"""
        self.setup_figure_style()
        grouped_data = dict(tuple(data.groupby('Frequency(GHz)')))
        colors = plt.cm.tab10(np.linspace(0, 1, len(grouped_data)))

        # 生成所有单独的图表
        self._plot_power_transfer(grouped_data, colors)
        self._plot_efficiency_vs_power(grouped_data, colors)
        self._plot_gain_vs_power(grouped_data, colors)
        self._plot_frequency_characteristics(grouped_data, colors)
        self._plot_combined_charts(grouped_data, colors)

    def _plot_power_transfer(self, grouped_data: Dict, colors: np.ndarray):
        """绘制功率传输特性图"""
        fig, ax = plt.subplots()
        
        for (freq, data), color in zip(grouped_data.items(), colors):
            input_col = 'Input_Power(dBm)' if 'Input_Power(dBm)' in data.columns else 'Source_Power(dBm)'
            ax.plot(data[input_col], data['Output_Power(dBm)'], 'o-',
                   label=f'{freq:.1f} GHz', color=color)
        
        ax.set_xlabel('Input Power (dBm)')
        ax.set_ylabel('Output Power (dBm)')
        ax.set_title('Power Transfer Characteristic')
        ax.legend()
        
        self.save_figure(fig, 'power_transfer')
        plt.close(fig)

    def _plot_efficiency_vs_power(self, grouped_data: Dict, colors: np.ndarray):
        """绘制效率vs输出功率图"""
        fig, ax = plt.subplots()
        
        for (freq, data), color in zip(grouped_data.items(), colors):
            ax.plot(data['Output_Power(dBm)'], data['Total_Efficiency(%)'], 'o-',
                   label=f'{freq:.1f} GHz', color=color)
        
        ax.set_xlabel('Output Power (dBm)')
        ax.set_ylabel('Total Efficiency (%)')
        ax.set_title('Total Efficiency vs Output Power')
        ax.legend()
        
        self.save_figure(fig, 'efficiency_vs_power')
        plt.close(fig)

    def _plot_gain_vs_power(self, grouped_data: Dict, colors: np.ndarray):
        """绘制增益vs输出功率图"""
        fig, ax = plt.subplots()
        
        for (freq, data), color in zip(grouped_data.items(), colors):
            ax.plot(data['Output_Power(dBm)'], data['Total_Gain(dB)'], 'o-',
                   label=f'{freq:.1f} GHz', color=color)
        
        ax.set_xlabel('Output Power (dBm)')
        ax.set_ylabel('Total Gain (dB)')
        ax.set_title('Total Gain vs Output Power')
        ax.legend()
        
        self.save_figure(fig, 'gain_vs_power')
        plt.close(fig)

    def _plot_frequency_characteristics(self, grouped_data: Dict, colors: np.ndarray):
        """绘制频率特性图"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        freqs = []
        efficiencies = []
        powers = []
        gains = []
        
        for freq, data in grouped_data.items():
            freqs.append(freq)
            efficiencies.append(data['Total_Efficiency(%)'].max())
            powers.append(data['Output_Power(dBm)'].max())
            gains.append(data['Total_Gain(dB)'].iloc[0])  # 小信号增益
        
        # 效率柱状图
        ax1.bar(freqs, efficiencies, color=colors)
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('Peak Efficiency (%)')
        ax1.set_title('Peak Efficiency vs Frequency')

        # 输出功率柱状图
        ax2.bar(freqs, powers, color=colors)
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('Saturated Power (dBm)')
        ax2.set_title('Saturated Power vs Frequency')

        # 增益柱状图
        ax3.bar(freqs, gains, color=colors)
        ax3.set_xlabel('Frequency (GHz)')
        ax3.set_ylabel('Small Signal Gain (dB)')
        ax3.set_title('Small Signal Gain vs Frequency')

        plt.tight_layout()
        self.save_figure(fig, 'frequency_characteristics')
        plt.close(fig)

    def _plot_combined_charts(self, grouped_data: Dict, colors: np.ndarray):
        """绘制组合图表"""
        # 双Y轴图
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        for (freq, data), color in zip(grouped_data.items(), colors):
            ax1.plot(data['Output_Power(dBm)'], data['Total_Efficiency(%)'], 'o-',
                    label=f'{freq:.1f} GHz (Eff)', color=color)
            ax2.plot(data['Output_Power(dBm)'], data['Total_Gain(dB)'], 's--',
                    label=f'{freq:.1f} GHz (Gain)', color=color, alpha=0.5)
        
        ax1.set_xlabel('Output Power (dBm)')
        ax1.set_ylabel('Total Efficiency (%)')
        ax2.set_ylabel('Total Gain (dB)')
        ax1.set_title('Total Efficiency and Gain vs Output Power')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', 
                  bbox_to_anchor=(1.15, 0.5))
        
        plt.tight_layout()
        self.save_figure(fig, 'combined_eff_gain')
        plt.close(fig)

    def generate_html_report(self, data: pd.DataFrame, summary: dict):
        """生成HTML测试报告"""
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath="./"),
            autoescape=True
        )
        
        template = env.from_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Doherty功放测试报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .chart { margin: 20px 0; text-align: center; }
                .chart img { max-width: 100%; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f5f5f5; }
                .summary { margin: 20px 0; }
                .header { text-align: center; margin-bottom: 40px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Doherty功放测试报告</h1>
                <p>测试时间: {{ timestamp }}</p>
            </div>
            
            <div class="summary">
                <h2>测试配置</h2>
                <table>
                    <tr><th>参数</th><th>值</th></tr>
                    <tr><td>压缩类型</td><td>{{ compression_type }}</td></tr>
                    <tr><td>驱动模式</td><td>{{ driver_mode }}</td></tr>
                    <tr><td>测试频率</td><td>{{ frequencies }}</td></tr>
                </table>
            </div>
            
            <div class="summary">
                <h2>测试结果总结</h2>
                <table>
                    <tr>
                        <th>频率 (GHz)</th>
                        <th>饱和输出功率 (dBm)</th>
                        <th>峰值效率 (%)</th>
                        <th>小信号增益 (dB)</th>
                    </tr>
                    {% for freq in summary.frequencies %}
                    <tr>
                        <td>{{ "%.1f"|format(freq) }}</td>
                        <td>{{ "%.1f"|format(summary.max_power[freq]) }}</td>
                        <td>{{ "%.1f"|format(summary.peak_efficiency[freq]) }}</td>
                        <td>{{ "%.1f"|format(summary.small_signal_gain[freq]) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="charts">
                <h2>测试图表</h2>
                {% for chart in charts %}
                <div class="chart">
                    <h3>{{ chart.title }}</h3>
                    <img src="{{ chart.path }}" alt="{{ chart.title }}">
                </div>
                {% endfor %}
            </div>
        </body>
        </html>
        """)
        
        # 准备报告数据
        report_data = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'compression_type': self.data_manager.config['test_config']['compression_type'],
            'driver_mode': '启用' if self.data_manager.config['test_config']['driver_mode'] else '禁用',
            'frequencies': ', '.join([f"{f/1e9:.1f} GHz" for f in self.data_manager.config['test_config']['frequencies']]),
            'summary': summary,
            'charts': [
                {'title': '功率传输特性', 'path': 'charts/power_transfer.png'},
                {'title': '效率vs输出功率', 'path': 'charts/efficiency_vs_power.png'},
                {'title': '增益vs输出功率', 'path': 'charts/gain_vs_power.png'},
                {'title': '频率特性', 'path': 'charts/frequency_characteristics.png'},
                {'title': '效率和增益组合图', 'path': 'charts/combined_eff_gain.png'}
            ]
        }
        
        # 生成报告
        html_content = template.render(**report_data)
        with open(self.data_manager.base_dir / 'report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

def calculate_summary(data: pd.DataFrame) -> dict:
    """计算测试数据总结"""
    grouped = data.groupby('Frequency(GHz)')
    frequencies = sorted(data['Frequency(GHz)'].unique())
    
    summary = {
        'frequencies': frequencies,
        'max_power': {},
        'peak_efficiency': {},
        'small_signal_gain': {}
    }
    
    for freq in frequencies:
        freq_data = grouped.get_group(freq)
        summary['max_power'][freq] = freq_data['Output_Power(dBm)'].max()
        summary['peak_efficiency'][freq] = freq_data['Total_Efficiency(%)'].max()
        summary['small_signal_gain'][freq] = freq_data['Total_Gain(dB)'].iloc[0]
    
    return summary

def main():
    """主程序入口"""
    try:
        # 创建测试执行器
        executor = TestExecutor('config.json')
        
        # 检查是否有恢复点
        if executor.load_recovery_point():
            logging.info("从恢复点恢复测试数据")
        
        # 创建进度显示线程
        def show_progress():
            while True:
                try:
                    progress = executor.progress_queue.get(timeout=1)
                    print(f"\r测试进度: {progress:.1f}%", end="")
                    if progress >= 100:
                        print()  # 换行
                        break
                except queue.Empty:
                    continue
        
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()
        
        # 执行测试
        logging.info("开始测试...")
        executor.run_test()
        
        # 等待进度显示线程结束
        progress_thread.join()
        
        # 清理恢复点
        executor.cleanup_recovery_point()
        
        # 处理数据和生成报告
        test_data = pd.DataFrame(executor.test_data)
        executor.data_manager.save_test_data(test_data)
        
        # 计算总结数据
        summary = calculate_summary(test_data)
        executor.data_manager.save_summary(summary)
        
        # 生成图表和报告
        logging.info("生成图表和报告...")
        visualizer = Visualizer(executor.data_manager)
        visualizer.generate_all_charts(test_data)
        visualizer.generate_html_report(test_data, summary)
        
        logging.info(f"测试完成，结果保存在: {executor.data_manager.base_dir}")
        
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main() 