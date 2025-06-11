"""Doherty功放管理模块"""

from typing import Dict, List, Optional, Tuple
import pyvisa
import time
import logging
from dataclasses import dataclass
from doherty_config import DohertlyConfig, AmplifierBias

@dataclass
class PowerMeasurement:
    """功率测量结果"""
    voltage: float
    current: float
    power: float

class DohertlyManager:
    """Doherty功放管理类"""
    
    def __init__(self, config: DohertlyConfig, resource_manager: pyvisa.ResourceManager):
        """
        初始化Doherty功放管理器
        
        Args:
            config: Doherty配置对象
            resource_manager: VISA资源管理器
        """
        self.config = config
        self.rm = resource_manager
        self.instruments = {}
        self.dut_sources = {}
        
    def connect_instruments(self) -> None:
        """连接所有仪器"""
        try:
            # 连接非 DUT 电源类仪器
            for name, address in self.config.config['instruments'].items():
                if name.startswith('dut_dc_source'):
                    continue  # DUT电源后面处理

                # 非驱动模式下不连接 power_meter
                if not self.config.config['test_config']['driver_mode'] and name == 'power_meter':
                    logging.info("非驱动模式，跳过连接 power_meter")
                    continue

                # 非驱动模式下不连接 driver_dc_source
                if not self.config.config['test_config']['driver_mode'] and name == 'driver_dc_source':
                    logging.info("非驱动模式，跳过连接 driver_dc_source")
                    continue

                self.instruments[name] = self.rm.open_resource(address)
                logging.info(f"已连接 {name}: {self.instruments[name].query('*IDN?').strip()}")

            # 连接 DUT 电源仪器
            dut_mapping = self.config.get_dut_source_mapping()
            for amp_name, address in dut_mapping.items():
                self.dut_sources[amp_name] = self.rm.open_resource(address)
                logging.info(f"已连接 {amp_name} 电源: {self.dut_sources[amp_name].query('*IDN?').strip()}")

        except Exception as e:
            logging.error(f"连接仪器失败: {e}")
            self.disconnect_all()
            raise
            
    def disconnect_all(self) -> None:
        """断开所有仪器连接"""
        # 关闭信号源
        if 'signal_gen' in self.instruments:
            try:
                self.instruments['signal_gen'].write('OUTP:STAT OFF')
            except Exception as e:
                logging.warning(f"关闭信号源失败: {e}")
                
        # 按顺序关闭DUT电源
        for source_name in reversed(self.config.get_required_sources()):
            if source_name in self.dut_sources:
                try:
                    source = self.dut_sources[source_name]
                    source.write('OUTP CH2,OFF')
                    time.sleep(0.1)
                    source.write('OUTP CH1,OFF')
                    source.close()
                except Exception as e:
                    logging.warning(f"关闭{source_name}电源失败: {e}")
                    
        # 关闭驱动功放电源
        if 'driver_dc_source' in self.instruments:
            try:
                driver_dc = self.instruments['driver_dc_source']
                driver_dc.write('OUTP CH2,OFF')
                time.sleep(0.1)
                driver_dc.write('OUTP CH1,OFF')
                driver_dc.close()
            except Exception as e:
                logging.warning(f"关闭驱动功放电源失败: {e}")
                
        # 关闭其他仪器
        for name, instrument in self.instruments.items():
            if name not in ['signal_gen', 'driver_dc_source']:
                try:
                    instrument.close()
                except Exception as e:
                    logging.warning(f"关闭{name}失败: {e}")
                    
        self.instruments.clear()
        self.dut_sources.clear()
        
    def setup_power_meter(self, freq_hz: float) -> None:
        """设置功率计"""
        if 'power_meter' not in self.instruments:
            return
            
        try:
            power_meter = self.instruments['power_meter']
            power_meter.write('*RST')
            power_meter.write(f'SENS:FREQ {freq_hz}')
            power_meter.write('INIT:CONT OFF')  # 单次触发模式
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"功率计设置失败: {e}")
            raise
            
    def setup_driver_bias(self) -> bool:
        """设置驱动功放偏置"""
        if 'driver_dc_source' not in self.instruments:
            return True
            
        try:
            dc = self.instruments['driver_dc_source']
            bias = self.config.get_bias_config('driver')
            
            # 设置栅极偏置（CH1）
            dc.write(':INST CH1')
            dc.write(':VOLTAGE:PROT:STAT OFF')
            dc.write(':CURR:PROT:STAT OFF')
            dc.write(f':APPL CH1,{bias.gate.voltage},{bias.gate.current_limit}')
            dc.write('OUTP CH1,ON')
            time.sleep(1)
            
            # 设置漏极偏置（CH2）
            dc.write(':INST CH2')
            dc.write(f':VOLTAGE:PROT {bias.drain.voltage + 2}')
            dc.write(':VOLTAGE:PROT:STAT ON')
            dc.write(f':CURR:PROT {bias.drain.current_limit + 0.1}')
            dc.write(':CURR:PROT:STAT ON')
            dc.write(f':APPL CH2,{bias.drain.voltage},{bias.drain.current_limit}')
            dc.write('OUTP CH2,ON')
            time.sleep(1)
            
            # 检查CC模式
            if dc.query(':OUTP:MODE? CH2').strip() == "CC":
                logging.error("驱动功放电源进入CC模式，请检查电路连接!")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"设置驱动功放偏置失败: {e}")
            raise
            
    def setup_doherty_bias(self) -> bool:
        """设置Doherty功放偏置"""
        try:
            # 按载波→峰值顺序设置偏置
            for source_name in self.config.get_required_sources():
                if not self._setup_single_amplifier_bias(source_name):
                    return False
            return True
            
        except Exception as e:
            logging.error(f"设置Doherty功放偏置失败: {e}")
            raise
            
    def _setup_single_amplifier_bias(self, amp_name: str) -> bool:
        """设置单个功放偏置"""
        try:
            dc = self.dut_sources[amp_name]
            bias = self.config.get_bias_config(amp_name)
            
            # 设置栅极偏置（CH1）
            dc.write(':INST CH1')
            dc.write(':VOLTAGE:PROT:STAT OFF')
            dc.write(':CURR:PROT:STAT OFF')
            dc.write(f':APPL CH1,{bias.gate.voltage},{bias.gate.current_limit}')
            dc.write('OUTP CH1,ON')
            time.sleep(1)
            
            # 检查静态电流（如果有设置）
            if bias.gate.quiescent_current is not None:
                i1 = float(dc.query(':MEAS:CURR? CH1'))
                if abs(i1 - bias.gate.quiescent_current) > 0.01:  # 10mA误差
                    logging.warning(f"{amp_name}静态电流{i1:.3f}A与设定值"
                                  f"{bias.gate.quiescent_current:.3f}A不符")
            
            # 设置漏极偏置（CH2）
            dc.write(':INST CH2')
            dc.write(f':VOLTAGE:PROT {bias.drain.voltage + 2}')
            dc.write(':VOLTAGE:PROT:STAT ON')
            dc.write(f':CURR:PROT {bias.drain.current_limit + 0.1}')
            dc.write(':CURR:PROT:STAT ON')
            dc.write(f':APPL CH2,{bias.drain.voltage},{bias.drain.current_limit}')
            dc.write('OUTP CH2,ON')
            time.sleep(1)
            
            # 检查CC模式
            if dc.query(':OUTP:MODE? CH2').strip() == "CC":
                logging.error(f"{amp_name}电源进入CC模式，请检查电路连接!")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"设置{amp_name}偏置失败: {e}")
            raise
            
    def read_input_power(self) -> Optional[float]:
        """读取功率计数值"""
        if 'power_meter' not in self.instruments:
            return None
            
        try:
            power_meter = self.instruments['power_meter']
            power_meter.write('INIT:IMM')
            time.sleep(0.1)
            return float(power_meter.query('FETCH?'))
        except Exception as e:
            logging.warning(f"读取功率计失败: {e}")
            return None
            
    def measure_amplifier_power(self, source_name: str) -> PowerMeasurement:
        """测量单个功放的功耗"""
        try:
            dc = self.dut_sources[source_name]
            
            v1 = float(dc.query(':MEAS:VOLT? CH1'))
            i1 = float(dc.query(':MEAS:CURR? CH1'))
            v2 = float(dc.query(':MEAS:VOLT? CH2'))
            i2 = float(dc.query(':MEAS:CURR? CH2'))
            
            total_power = v1 * i1 + v2 * i2
            return PowerMeasurement(
                voltage=(v1, v2),
                current=(i1, i2),
                power=total_power
            )
            
        except Exception as e:
            logging.error(f"测量{source_name}功耗失败: {e}")
            raise
            
    def measure_total_power(self) -> Dict[str, PowerMeasurement]:
        """测量所有功放的功耗"""
        measurements = {}
        
        # 测量驱动功放功耗
        if 'driver_dc_source' in self.instruments:
            try:
                dc = self.instruments['driver_dc_source']
                v1 = float(dc.query(':MEAS:VOLT? CH1'))
                i1 = float(dc.query(':MEAS:CURR? CH1'))
                v2 = float(dc.query(':MEAS:VOLT? CH2'))
                i2 = float(dc.query(':MEAS:CURR? CH2'))
                measurements['driver'] = PowerMeasurement(
                    voltage=(v1, v2),
                    current=(i1, i2),
                    power=v1*i1 + v2*i2
                )
            except Exception as e:
                logging.warning(f"测量驱动功放功耗失败: {e}")
                
        # 测量各路DUT功耗
        for source_name in self.config.get_required_sources():
            try:
                measurements[source_name] = self.measure_amplifier_power(source_name)
            except Exception as e:
                logging.warning(f"测量{source_name}功耗失败: {e}")
                
        return measurements
        
    def calculate_efficiency(self, rf_output_power: float) -> Dict[str, float]:
        """计算各路功放效率"""
        measurements = self.measure_total_power()
        rf_output_watt = 10 ** ((rf_output_power - 30) / 10)
        
        efficiencies = {}
        total_dc_power = 0
        
        # 仅计算 driver 的效率
        if 'driver' in measurements:
            # 驱动功放使用功率计读数作为输出功率
            output_power = self.read_input_power()
            if output_power is not None:
                output_watt = 10 ** ((output_power - 30) / 10)
                efficiencies['driver'] = output_watt / measurements['driver'].power * 100
            else:
                efficiencies['driver'] = 0.0
        # 只统计 DUT 总功率用于 total 效率计算
        total_dc_power = sum(
            m.power for name, m in measurements.items() if name != 'driver'
        )
            
        # 计算总效率
        efficiencies['total'] = (rf_output_watt / total_dc_power * 100) if total_dc_power > 0 else 0
        
        return efficiencies 