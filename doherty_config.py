"""Doherty功放配置验证模块"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import json
import logging
from enum import Enum
from pathlib import Path

class DohertlyType(Enum):
    """Doherty功放类型"""
    TWO_WAY = "2-way"
    THREE_WAY = "3-way"
    SYMMETRIC = "symmetric"

@dataclass
class BiasConfig:
    """偏置配置数据类"""
    voltage: float
    current_limit: float
    quiescent_current: Optional[float] = None

@dataclass
class AmplifierBias:
    """功放偏置配置"""
    gate: BiasConfig
    drain: BiasConfig

class DohertlyConfig:
    """Doherty功放测试系统配置管理类"""
    def __init__(self, config_file: str):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.validate_config()
        
    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"加载配置文件失败: {e}")
            
    def validate_config(self):
        """验证配置文件的完整性和正确性"""
        required_sections = [
            'test_config', 'instruments', 'bias_settings', 
            'doherty_configurations'
        ]
        
        # 检查必要的配置部分
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件缺少必要的{section}部分")
                
        # 验证测试配置
        test_config = self.config['test_config']
        self._validate_test_config(test_config)
        
        # 验证仪器配置
        self._validate_instruments_config()
        
        # 验证Doherty配置
        self._validate_doherty_config()
        
    def _validate_test_config(self, test_config: dict):
        """验证测试配置"""
        required_fields = ['frequencies', 'power_range', 'compression_type', 
                         'driver_mode', 'doherty_type']
                         
        for field in required_fields:
            if field not in test_config:
                raise ValueError(f"测试配置缺少{field}字段")
                
        # 验证功率范围设置
        power_range = test_config['power_range']
        if not all(k in power_range for k in ['start', 'stop', 'step']):
            raise ValueError("功率范围配置不完整")
            
        if power_range['start'] >= power_range['stop']:
            raise ValueError("功率范围起始值必须小于结束值")
            
        # 验证压缩点类型
        if test_config['compression_type'] not in ['1dB', '3dB']:
            raise ValueError("压缩点类型必须是'1dB'或'3dB'")
            
    def _validate_instruments_config(self):
        """验证仪器配置"""
        required_instruments = ['spectrum_analyzer', 'signal_gen']
        if self.config['test_config']['driver_mode']:
            required_instruments.append('driver_dc_source')
            required_instruments.append('power_meter')
            
        # 根据Doherty类型确定所需的电源数量
        doherty_type = self.config['test_config']['doherty_type']
        doherty_config = self.config['doherty_configurations'][doherty_type]
        required_dc_count = doherty_config['required_dc_sources']
        
        for i in range(1, required_dc_count + 1):
            required_instruments.append(f'dut_dc_source_{i}')
            
        for instrument in required_instruments:
            if instrument not in self.config['instruments']:
                raise ValueError(f"缺少必要的仪器配置: {instrument}")
                
    def _validate_doherty_config(self):
        """验证Doherty配置"""
        doherty_type = self.config['test_config']['doherty_type']
        if doherty_type not in self.config['doherty_configurations']:
            raise ValueError(f"未找到Doherty类型{doherty_type}的配置")
            
        # 验证偏置设置
        doherty_config = self.config['doherty_configurations'][doherty_type]
        for source in doherty_config['power_sources']:
            if source not in self.config['bias_settings']:
                raise ValueError(f"未找到功放{source}的偏置设置")
                
    def get_required_sources(self) -> List[str]:
        """获取当前配置需要的功放电源列表"""
        doherty_type = self.config['test_config']['doherty_type']
        return self.config['doherty_configurations'][doherty_type]['power_sources']

    def get_dut_source_mapping(self) -> Dict[str, str]:
        """
        将 power_sources（如 carrier_amp） 映射到实际仪器名（如 dut_dc_source_1）地址
        """
        sources = self.get_required_sources()
        mapping = {}
        for i, amp_name in enumerate(sources, start=1):
            inst_key = f'dut_dc_source_{i}'
            try:
                mapping[amp_name] = self.config['instruments'][inst_key]
            except KeyError:
                raise ValueError(f"在 instruments 中找不到键 {inst_key}，请检查配置文件")
        return mapping

    def get_bias_settings(self, amp_name: str) -> Dict:
        """获取指定功放的偏置设置"""
        return self.config['bias_settings'][amp_name]
        
    def get_instrument_address(self, name: str) -> str:
        """获取仪器地址"""
        return self.config['instruments'].get(name)
        
    def get_test_frequencies(self) -> List[float]:
        """获取测试频率列表"""
        return self.config['test_config']['frequencies']
        
    def get_power_range(self) -> Dict:
        """获取功率扫描范围设置"""
        return self.config['test_config']['power_range']
        
    def is_driver_mode_enabled(self) -> bool:
        """检查是否启用驱动模式"""
        return self.config['test_config']['driver_mode']
        
    def get_compression_type(self) -> str:
        """获取压缩点类型"""
        return self.config['test_config']['compression_type']
        
    def get_doherty_type(self) -> str:
        """获取Doherty类型"""
        return self.config['test_config']['doherty_type']
        
    def get_bias_config(self, amp_type: str) -> AmplifierBias:
        """获取指定功放的偏置配置"""
        if amp_type == 'driver':
            bias = self.config['bias_config']['driver']
        else:
            bias = self.config['bias_config']['dut'][amp_type]
            
        return AmplifierBias(
            gate=BiasConfig(**bias['gate']),
            drain=BiasConfig(**bias['drain'])
        )
        
    def is_symmetric(self) -> bool:
        """检查是否为对称Doherty"""
        return self.config['test_config']['doherty_type'] == DohertlyType.SYMMETRIC.value
