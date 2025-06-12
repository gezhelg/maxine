#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pyvisa
import numpy as np
import pandas as pd
import time
import datetime
import logging
from pathlib import Path
import shutil
import jinja2
from typing import Dict, List

# Import the well-defined modules
from doherty_config import DohertyConfig
from doherty_manager import DohertyManager
from visualization import PAVisualization

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('doherty_test.log')
    ]
)

class DataManager:
    """
    Manages all data and file operations, including saving results,
    configuration, and generating reports.
    Responsibility: Data Persistence and File I/O.
    """
    def __init__(self, config: DohertyConfig):
        self.config_data = config.config
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(self.config_data['output_config']['base_dir']) / self.timestamp
        self.charts_dir = self.base_dir / 'charts'
        self.backup_dir = Path(self.config_data['output_config']['backup_dir'])
        self._setup_directories()

    def _setup_directories(self):
        """Creates the necessary output and backup directories."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Results will be saved in: {self.base_dir}")

    def save_results(self, data: pd.DataFrame, summary: dict):
        """Saves all test results and a copy of the config."""
        # Save raw data
        results_path = self.base_dir / 'test_results.csv'
        data.to_csv(results_path, index=False)
        self._backup_file(results_path)
        
        # Save summary
        summary_path = self.base_dir / 'summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, default=str)
        self._backup_file(summary_path)

        # Save the config used for this test run
        config_path = self.base_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config_data, f, indent=4)
        self._backup_file(config_path)

    def _backup_file(self, file_path: Path):
        """Backs up a given file to the backup directory."""
        try:
            backup_path = self.backup_dir / f"{self.timestamp}_{file_path.name}"
            shutil.copy2(file_path, backup_path)
        except Exception as e:
            logging.warning(f"Could not back up {file_path.name}: {e}")

class TestController:
    """
    Orchestrates the entire test process, from setup to execution and
    finally to data processing and visualization.
    Responsibility: Test Flow Orchestration.
    """
    def __init__(self, config_file: str):
        """Initializes all necessary components for the test."""
        self.config = DohertyConfig(config_file)
        self.manager = DohertyManager(self.config, pyvisa.ResourceManager())
        self.data_handler = DataManager(self.config)
        self.visualizer = PAVisualization(save_dir=str(self.data_handler.charts_dir))
        self.test_results = []

    def run_full_test(self):
        """
        Executes the complete test suite: instrument setup, frequency sweep,
        data processing, and report generation.
        """
        try:
            # Connect and setup instruments once
            self.manager.connect_instruments()

            frequencies = self.config.get_test_frequencies()
            total_freqs = len(frequencies)

            for i, freq in enumerate(frequencies, 1):
                logging.info(f"--- Testing Frequency {freq/1e6:.1f} MHz ({i}/{total_freqs}) ---")
                freq_results = self._run_sweep_for_frequency(freq)
                if freq_results:
                    self.test_results.extend(freq_results)

            # Process and save all collected data
            if not self.test_results:
                logging.warning("No data was collected during the test.")
                return

            results_df = pd.DataFrame(self.test_results)
            summary = self._calculate_summary(results_df)
            self.data_handler.save_results(results_df, summary)
            
            # Generate visualizations
            logging.info("Generating charts and reports...")
            self._generate_visualizations(results_df)

        except Exception as e:
            logging.error(f"A critical error occurred during the test: {e}", exc_info=True)
        finally:
            logging.info("Cleaning up and disconnecting instruments.")
            self.cleanup()
            
    def _run_sweep_for_frequency(self, freq: float) -> List[dict]:
        """
        Performs a power sweep for a single frequency point.
        
        Returns:
            A list of dictionary objects, each containing the results for one power step.
        """
        results = []
        try:
            # Setup instruments for the specific frequency
            self.manager.setup_power_meter(freq)
            if not self.manager.setup_driver_bias() or not self.manager.setup_doherty_bias():
                raise RuntimeError(f"Bias setup failed for frequency {freq/1e6:.1f} MHz.")

            # Configure signal generator and spectrum analyzer
            signal_gen = self.manager.instruments['signal_gen']
            signal_gen.write(f'FREQ {freq}Hz')
            signal_gen.write('OUTP:STAT ON')

            spectrum = self.manager.instruments['spectrum_analyzer']
            spectrum.write(f'FREQ:CENT {freq}Hz')
            spectrum.write('FREQ:SPAN 400MHz') # Example span, can be configured
            spectrum.write('CALC:MARK:MAX')

            power_range = self.config.get_power_range()
            power_steps = np.arange(power_range['start'], power_range['stop'] + power_range['step'], power_range['step'])

            for power_dbm in power_steps:
                signal_gen.write(f'POW {power_dbm:.2f}dBm')
                time.sleep(0.2)  # Settling time

                # Perform measurements
                input_power = self.manager.read_input_power() if self.config.is_driver_mode_enabled() else power_dbm
                output_power = float(spectrum.query('CALC:MARK:Y?'))
                
                power_consumption = self.manager.measure_total_power()
                total_dc_power = sum(m.power for name, m in power_consumption.items() if name != 'driver')
                
                # Calculate metrics
                gain = output_power - input_power
                output_power_watts = 10**((output_power - 30) / 10)
                efficiency = (output_power_watts / total_dc_power * 100) if total_dc_power > 0 else 0

                # Log and store result
                logging.info(f"Pin: {input_power:.2f} dBm, Pout: {output_power:.2f} dBm, Gain: {gain:.2f} dB, Eff: {efficiency:.2f}%")
                
                result_row = {
                    'frequency_hz': freq,
                    'input_power_dbm': input_power,
                    'output_power_dbm': output_power,
                    'gain_db': gain,
                    'efficiency_percent': efficiency,
                    'total_dc_power_w': total_dc_power,
                }
                # Add detailed power consumption data
                for amp, data in power_consumption.items():
                    result_row[f'{amp}_dc_power_w'] = data.power
                
                results.append(result_row)
        
        except Exception as e:
            logging.error(f"Failed during sweep for {freq/1e6:.1f} MHz: {e}", exc_info=True)
            return [] # Return empty list on failure for this frequency

        return results

    def _calculate_summary(self, data: pd.DataFrame) -> dict:
        """Calculates key performance indicators from the test data."""
        summary = {'by_frequency': {}}
        grouped = data.groupby('frequency_hz')
        
        for freq, group in grouped:
            peak_efficiency_row = group.loc[group['efficiency_percent'].idxmax()]
            max_power_row = group.loc[group['output_power_dbm'].idxmax()]
            
            summary['by_frequency'][f"{freq/1e6:.1f}MHz"] = {
                'peak_efficiency_percent': peak_efficiency_row['efficiency_percent'],
                'power_at_peak_efficiency_dbm': peak_efficiency_row['output_power_dbm'],
                'saturated_output_power_dbm': max_power_row['output_power_dbm'],
                'gain_at_saturation_db': max_power_row['gain_db'],
                'small_signal_gain_db': group['gain_db'].iloc[0] if not group.empty else 0,
            }
        return summary

    def _transform_data_for_visualization(self, data: pd.DataFrame) -> Dict:
        """Converts the DataFrame into the nested dictionary format required by PAVisualization."""
        vis_data = {}
        grouped = data.groupby('frequency_hz')
        for freq, group in grouped:
            vis_data[freq] = {
                'pin': group['input_power_dbm'].values,
                'pout': group['output_power_dbm'].values,
                'efficiency': group['efficiency_percent'].values / 100.0, # Convert back to ratio
            }
        return vis_data

    def _generate_visualizations(self, data: pd.DataFrame):
        """Uses the PAVisualization module to create and save all plots."""
        vis_data = self._transform_data_for_visualization(data)
        
        if not vis_data:
            logging.warning("No data available to generate visualizations.")
            return
            
        # Use the methods from the dedicated visualization module
        self.visualizer.plot_power_sweep(vis_data)
        self.visualizer.plot_efficiency_vs_pout(vis_data)
        self.visualizer.plot_gain_vs_pout(vis_data)
        self.visualizer.plot_frequency_summary(vis_data)
        self.visualizer.plot_combined_efficiency_gain(vis_data)
        self.visualizer.generate_summary_report(vis_data)
        logging.info("All charts and summary report have been generated.")

    def cleanup(self):
        """Disconnects all instruments safely."""
        self.manager.disconnect_all()

def main():
    """Main program entry point."""
    config_file = 'config.json'  # Assume config file is in the same directory
    
    # Check if config file exists
    if not Path(config_file).exists():
        logging.error(f"Configuration file '{config_file}' not found. Aborting.")
        return

    logging.info("Initializing Doherty PA Test Controller...")
    controller = TestController(config_file)
    
    logging.info("Starting test sequence...")
    controller.run_full_test()
    
    logging.info("Test sequence finished.")

if __name__ == "__main__":
    main()