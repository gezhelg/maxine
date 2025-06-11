import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns

class PAVisualization:
    def __init__(self, save_dir: str = "results/plots"):
        """Initialize the visualization module.
        
        Args:
            save_dir: Directory to save plots and data files
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn')
        self.colors = sns.color_palette("husl", 8)
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
    def _setup_plot(self, title: str, xlabel: str, ylabel: str) -> Tuple[plt.Figure, plt.Axes]:
        """Setup a new plot with common formatting."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(title, pad=20, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        return fig, ax

    def _save_plot(self, fig: plt.Figure, name: str):
        """Save plot in both PNG and PDF formats."""
        fig.tight_layout()
        fig.savefig(self.save_dir / f"{name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(self.save_dir / f"{name}.pdf", bbox_inches='tight')
        plt.close(fig)

    def find_saturation_point(self, pin: np.ndarray, pout: np.ndarray, 
                            compression_db: float = 1.0) -> Tuple[float, float]:
        """Find the saturation point based on gain compression."""
        gain = 10 * np.log10(pout / pin)
        max_gain = np.max(gain)
        compression_point = max_gain - compression_db
        
        # Find the point closest to compression_point
        idx = np.abs(gain - compression_point).argmin()
        return pin[idx], pout[idx]

    def plot_power_sweep(self, data: Dict[float, Dict[str, np.ndarray]]):
        """Plot input power vs output power for different frequencies."""
        fig, ax = self._setup_plot(
            "Power Sweep at Different Frequencies",
            "Input Power (dBm)",
            "Output Power (dBm)"
        )
        
        for idx, (freq, measurements) in enumerate(data.items()):
            pin = measurements['pin']
            pout = measurements['pout']
            
            # Plot data points
            ax.plot(pin, pout, 
                   marker=self.markers[idx % len(self.markers)],
                   color=self.colors[idx % len(self.colors)],
                   label=f"{freq/1e6:.0f} MHz",
                   markersize=6)
            
            # Mark saturation point
            sat_pin, sat_pout = self.find_saturation_point(pin, pout)
            ax.plot(sat_pin, sat_pout, 'x', color=self.colors[idx % len(self.colors)],
                   markersize=10, markeredgewidth=2)
            
        ax.legend(title="Frequency", bbox_to_anchor=(1.05, 1), loc='upper left')
        self._save_plot(fig, "power_sweep")
        
        # Export data to CSV
        self._export_data(data, "power_sweep")

    def plot_efficiency_vs_pout(self, data: Dict[float, Dict[str, np.ndarray]]):
        """Plot drain efficiency vs output power for different frequencies."""
        fig, ax = self._setup_plot(
            "Drain Efficiency vs Output Power",
            "Output Power (dBm)",
            "Drain Efficiency (%)"
        )
        
        for idx, (freq, measurements) in enumerate(data.items()):
            pout = measurements['pout']
            efficiency = measurements['efficiency'] * 100  # Convert to percentage
            
            ax.plot(pout, efficiency,
                   marker=self.markers[idx % len(self.markers)],
                   color=self.colors[idx % len(self.colors)],
                   label=f"{freq/1e6:.0f} MHz")
            
        ax.legend(title="Frequency", bbox_to_anchor=(1.05, 1), loc='upper left')
        self._save_plot(fig, "efficiency_vs_pout")
        
        # Export data to CSV
        self._export_data(data, "efficiency_vs_pout")

    def plot_gain_vs_pout(self, data: Dict[float, Dict[str, np.ndarray]]):
        """Plot gain vs output power for different frequencies."""
        fig, ax = self._setup_plot(
            "Power Gain vs Output Power",
            "Output Power (dBm)",
            "Power Gain (dB)"
        )
        
        for idx, (freq, measurements) in enumerate(data.items()):
            pin = measurements['pin']
            pout = measurements['pout']
            gain = 10 * np.log10(measurements['pout'] / measurements['pin'])
            
            ax.plot(pout, gain,
                   marker=self.markers[idx % len(self.markers)],
                   color=self.colors[idx % len(self.colors)],
                   label=f"{freq/1e6:.0f} MHz")
            
        ax.legend(title="Frequency", bbox_to_anchor=(1.05, 1), loc='upper left')
        self._save_plot(fig, "gain_vs_pout")
        
        # Export data to CSV
        self._export_data(data, "gain_vs_pout")

    def plot_frequency_summary(self, data: Dict[float, Dict[str, np.ndarray]]):
        """Create combined plots showing frequency response characteristics."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        freqs = np.array(list(data.keys())) / 1e6  # Convert to MHz
        sat_power = []
        sat_efficiency = []
        sat_gain = []
        
        for freq, measurements in data.items():
            sat_pin, sat_pout = self.find_saturation_point(
                measurements['pin'], measurements['pout'])
            idx = np.where(measurements['pin'] == sat_pin)[0][0]
            
            sat_power.append(sat_pout)
            sat_efficiency.append(measurements['efficiency'][idx] * 100)
            sat_gain.append(10 * np.log10(measurements['pout'][idx] / measurements['pin'][idx]))
        
        # Plot 1: Saturated Output Power
        ax1.bar(freqs, sat_power, color=self.colors[0])
        ax1.set_title("Saturated Output Power vs Frequency")
        ax1.set_ylabel("Output Power (dBm)")
        
        # Plot 2: Drain Efficiency at Saturation
        ax2.bar(freqs, sat_efficiency, color=self.colors[1])
        ax2.set_title("Drain Efficiency at Saturation vs Frequency")
        ax2.set_ylabel("Drain Efficiency (%)")
        
        # Plot 3: Gain at Saturation
        ax3.bar(freqs, sat_gain, color=self.colors[2])
        ax3.set_title("Power Gain at Saturation vs Frequency")
        ax3.set_xlabel("Frequency (MHz)")
        ax3.set_ylabel("Power Gain (dB)")
        
        plt.tight_layout()
        self._save_plot(fig, "frequency_summary")
        
        # Export summary data
        summary_data = pd.DataFrame({
            'Frequency_MHz': freqs,
            'Saturated_Power_dBm': sat_power,
            'Saturated_Efficiency_Percent': sat_efficiency,
            'Saturated_Gain_dB': sat_gain
        })
        summary_data.to_csv(self.save_dir / "frequency_summary.csv", index=False)

    def plot_combined_efficiency_gain(self, data: Dict[float, Dict[str, np.ndarray]]):
        """Create combined plot with efficiency and gain on dual Y-axes."""
        for freq, measurements in data.items():
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()
            
            pout = measurements['pout']
            efficiency = measurements['efficiency'] * 100
            gain = 10 * np.log10(measurements['pout'] / measurements['pin'])
            
            # Plot efficiency
            line1 = ax1.plot(pout, efficiency, color=self.colors[0], 
                           marker='o', label='Efficiency')
            ax1.set_xlabel("Output Power (dBm)")
            ax1.set_ylabel("Drain Efficiency (%)", color=self.colors[0])
            ax1.tick_params(axis='y', labelcolor=self.colors[0])
            
            # Plot gain
            line2 = ax2.plot(pout, gain, color=self.colors[1], 
                           marker='s', label='Gain')
            ax2.set_ylabel("Power Gain (dB)", color=self.colors[1])
            ax2.tick_params(axis='y', labelcolor=self.colors[1])
            
            # Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')
            
            plt.title(f"Efficiency and Gain vs Output Power at {freq/1e6:.0f} MHz")
            self._save_plot(fig, f"combined_eff_gain_{freq/1e6:.0f}MHz")

    def _export_data(self, data: Dict[float, Dict[str, np.ndarray]], name: str):
        """Export measurement data to CSV files."""
        for freq, measurements in data.items():
            df = pd.DataFrame(measurements)
            df['frequency'] = freq
            df.to_csv(self.save_dir / f"{name}_{freq/1e6:.0f}MHz.csv", index=False)

    def generate_summary_report(self, data: Dict[float, Dict[str, np.ndarray]]) -> str:
        """Generate a summary report of the measurements."""
        report = []
        report.append("Power Amplifier Test Summary Report")
        report.append("=" * 40)
        report.append("")
        
        for freq, measurements in data.items():
            sat_pin, sat_pout = self.find_saturation_point(
                measurements['pin'], measurements['pout'])
            idx = np.where(measurements['pin'] == sat_pin)[0][0]
            
            report.append(f"Frequency: {freq/1e6:.1f} MHz")
            report.append("-" * 20)
            report.append(f"Saturated Output Power: {sat_pout:.1f} dBm")
            report.append(f"Drain Efficiency at Saturation: {measurements['efficiency'][idx]*100:.1f}%")
            report.append(f"Power Gain at Saturation: {10*np.log10(measurements['pout'][idx]/measurements['pin'][idx]):.1f} dB")
            report.append(f"Maximum Drain Efficiency: {np.max(measurements['efficiency'])*100:.1f}%")
            report.append("")
        
        report_text = "\n".join(report)
        with open(self.save_dir / "test_summary.txt", "w") as f:
            f.write(report_text)
            
        return report_text 