import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks, welch
import zipfile
import os
from pathlib import Path

class HeartRateFFTAnalyzer:
    def __init__(self, data_path="data/Nathaniel/"):
        self.data_path = data_path
        self.datasets = {}
        
    def load_all_data(self):
        """Load both no caffeine and caffeine datasets"""
        zip_files = {
            "no_caffeine": "No_caffeine-2025-06-04_13-45-27.zip",
            "caffeine": "30min_20mg_caffeine-2025-06-04_14-17-49.zip"
        }
        
        for condition, zip_name in zip_files.items():
            zip_path = os.path.join(self.data_path, zip_name)
            self.datasets[condition] = self._extract_sensor_data(zip_path)
            print(f"Loaded {condition} data: {len(self.datasets[condition])} files")
    
    def _extract_sensor_data(self, zip_path):
        """Extract and parse sensor data from zip file"""
        data_dict = {}
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.csv'):
                    try:
                        with zip_ref.open(file_name) as file:
                            df = pd.read_csv(file)
                            # Clean filename for easier access
                            clean_name = file_name.replace('.csv', '').split('/')[-1]
                            data_dict[clean_name] = df
                            print(f"  Loaded {clean_name}: shape {df.shape}")
                    except Exception as e:
                        print(f"  Error loading {file_name}: {e}")
        
        return data_dict
    
    def explore_data_structure(self, condition="no_caffeine"):
        """Explore the structure of sensor data to understand timestamps"""
        print(f"\n=== DATA STRUCTURE EXPLORATION ({condition}) ===")
        
        dataset = self.datasets[condition]
        
        for sensor_name, df in dataset.items():
            if sensor_name in ['WatchAccelerometer', 'WatchGyroscope', 'WatchTotalAcceleration', 'HeartRate']:
                print(f"\n{sensor_name}:")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Data types:\n{df.dtypes}")
                print(f"  First 3 rows:")
                print(df.head(3))
                
                # Look for time-related columns
                time_cols = [col for col in df.columns if any(word in col.lower() 
                            for word in ['time', 'timestamp', 'millis', 'nano', 'epoch'])]
                if time_cols:
                    print(f"  Time columns found: {time_cols}")
                    for time_col in time_cols:
                        print(f"    {time_col} sample values: {df[time_col].head(3).tolist()}")
                        print(f"    {time_col} range: {df[time_col].min()} to {df[time_col].max()}")
    
    def estimate_sampling_rate_improved(self, df):
        """Improved sampling rate estimation with multiple methods"""
        # Method 1: Look for explicit time columns
        time_cols = [col for col in df.columns if any(word in col.lower() 
                    for word in ['time', 'timestamp', 'millis', 'nano', 'epoch'])]
        
        for time_col in time_cols:
            try:
                times = pd.to_numeric(df[time_col], errors='coerce').dropna()
                if len(times) > 1:
                    time_diffs = np.diff(times)
                    time_diffs = time_diffs[time_diffs > 0]  # Remove zero differences
                    
                    if len(time_diffs) > 0:
                        median_interval = np.median(time_diffs)
                        
                        # Determine if milliseconds, microseconds, nanoseconds, or seconds
                        if median_interval > 1e6:  # Nanoseconds
                            sampling_rate = 1e9 / median_interval
                            print(f"  Detected nanosecond timestamps, sampling rate: {sampling_rate:.2f} Hz")
                        elif median_interval > 1000:  # Milliseconds or microseconds
                            sampling_rate = 1000.0 / median_interval
                            print(f"  Detected millisecond timestamps, sampling rate: {sampling_rate:.2f} Hz")
                        elif median_interval > 1:  # Large numbers, likely milliseconds
                            sampling_rate = 1000.0 / median_interval
                            print(f"  Assuming millisecond timestamps, sampling rate: {sampling_rate:.2f} Hz")
                        else:  # Seconds
                            sampling_rate = 1.0 / median_interval
                            print(f"  Detected second timestamps, sampling rate: {sampling_rate:.2f} Hz")
                        
                        if 1 <= sampling_rate <= 1000:  # Reasonable range
                            return sampling_rate
            except:
                continue
        
        # Method 2: Estimate from data length and typical measurement duration (30 seconds)
        n_samples = len(df)
        estimated_duration = 30.0  # seconds
        estimated_rate = n_samples / estimated_duration
        print(f"  Fallback estimate: {n_samples} samples / {estimated_duration}s = {estimated_rate:.2f} Hz")
        
        if 10 <= estimated_rate <= 200:  # Reasonable for smartwatch
            return estimated_rate
        
        # Method 3: Common smartwatch sampling rates
        common_rates = [50, 100, 25, 20]
        print(f"  Using common rate assumption: 50 Hz")
        return 50.0
    
    def extract_heart_rate_from_motion(self, motion_data, sampling_rate, hr_range=(0.8, 3.5)):
        """
        Extract heart rate from motion data using FFT
        hr_range: Expected heart rate range in Hz (48-210 BPM = 0.8-3.5 Hz)
        """
        results = {}
        
        print(f"  Processing with sampling rate: {sampling_rate:.2f} Hz")
        
        # Get numeric columns (motion sensors)
        numeric_cols = motion_data.select_dtypes(include=[np.number]).columns
        time_cols = [col for col in motion_data.columns if any(word in col.lower() 
                    for word in ['time', 'timestamp', 'millis', 'nano', 'epoch'])]
        
        # Exclude time columns from analysis
        sensor_cols = [col for col in numeric_cols if col not in time_cols]
        
        print(f"  Analyzing columns: {sensor_cols}")
        
        for col in sensor_cols:
            signal_data = motion_data[col].dropna().values
            
            if len(signal_data) < 100:  # Need sufficient data points
                print(f"    {col}: Insufficient data ({len(signal_data)} points)")
                continue
            
            # Remove DC component and detrend
            signal_data = signal_data - np.mean(signal_data)
            signal_data = signal.detrend(signal_data)
            
            # Check if we have reasonable sampling rate for filtering
            nyquist = sampling_rate / 2
            low_cut = hr_range[0] / nyquist
            high_cut = hr_range[1] / nyquist
            
            if low_cut >= 0.99 or high_cut >= 0.99:
                print(f"    {col}: Sampling rate too low for heart rate filtering")
                continue
            
            try:
                # Apply bandpass filter to heart rate range
                b, a = butter(4, [max(0.01, low_cut), min(0.99, high_cut)], btype='band')
                filtered_signal = filtfilt(b, a, signal_data)
                
                # Perform FFT
                n_samples = len(filtered_signal)
                freqs = fftfreq(n_samples, 1/sampling_rate)
                fft_vals = np.abs(fft(filtered_signal))
                
                # Only look at positive frequencies in heart rate range
                positive_freqs = freqs[:n_samples//2]
                positive_fft = fft_vals[:n_samples//2]
                
                hr_mask = (positive_freqs >= hr_range[0]) & (positive_freqs <= hr_range[1])
                hr_freqs = positive_freqs[hr_mask]
                hr_fft = positive_fft[hr_mask]
                
                if len(hr_freqs) > 0:
                    # Find peak frequency (dominant heart rate)
                    peak_idx = np.argmax(hr_fft)
                    peak_freq = hr_freqs[peak_idx]
                    peak_bpm = peak_freq * 60
                    peak_power = hr_fft[peak_idx]
                    
                    # Test for heart rate doubling artifact
                    peak_bpm_half = peak_bpm / 2
                    
                    # Determine which estimate is more physiologically reasonable
                    # Normal resting HR: 50-120 BPM, caffeine can increase to ~130 BPM
                    if 45 <= peak_bpm_half <= 130 and peak_bpm > 100:
                        # If half the rate is reasonable and original is high, use half
                        corrected_bpm = peak_bpm_half
                        correction_applied = True
                        print(f"      Applied halving correction: {peak_bpm:.1f} → {corrected_bpm:.1f} BPM")
                    else:
                        # Keep original estimate
                        corrected_bpm = peak_bpm
                        correction_applied = False
                    
                    # Calculate spectral features
                    total_power = np.sum(hr_fft**2)
                    peak_ratio = peak_power**2 / total_power if total_power > 0 else 0
                    
                    # Additional confidence metrics
                    # Find all peaks in the spectrum
                    peaks, _ = find_peaks(hr_fft, height=peak_power*0.3)
                    n_significant_peaks = len(peaks)
                    
                    results[col] = {
                        'peak_frequency_hz': peak_freq,
                        'estimated_bpm': peak_bpm,
                        'corrected_bpm': corrected_bpm,
                        'correction_applied': correction_applied,
                        'peak_power': peak_power,
                        'peak_ratio': peak_ratio,
                        'n_peaks': n_significant_peaks,
                        'frequencies': hr_freqs,
                        'fft_values': hr_fft,
                        'filtered_signal': filtered_signal
                    }
                    
                    print(f"    {col}: {corrected_bpm:.1f} BPM (original: {peak_bpm:.1f}, confidence: {peak_ratio:.3f}, peaks: {n_significant_peaks})")
                
            except Exception as e:
                print(f"    {col}: Processing error - {e}")
                continue
        
        return results
    
    def analyze_heart_rate_extraction(self, condition="no_caffeine"):
        """Analyze heart rate extraction for one condition"""
        if condition not in self.datasets:
            print(f"Condition {condition} not found")
            return None
            
        dataset = self.datasets[condition]
        results = {}
        
        # Get actual heart rate data if available
        actual_hr = None
        if 'HeartRate' in dataset:
            actual_hr = dataset['HeartRate']
            print(f"Found actual heart rate data: {actual_hr.shape}")
            print(f"Actual HR columns: {list(actual_hr.columns)}")
            
        print(f"\nAnalyzing {condition} condition:")
        print("="*40)
        
        # Analyze each motion sensor
        motion_sensors = ['WatchAccelerometer', 'WatchGyroscope', 'WatchTotalAcceleration']
        
        for sensor_name in motion_sensors:
            if sensor_name in dataset:
                print(f"\nProcessing {sensor_name}...")
                sensor_data = dataset[sensor_name]
                
                # Improved sampling rate estimation
                sampling_rate = self.estimate_sampling_rate_improved(sensor_data)
                
                # Extract heart rate
                hr_results = self.extract_heart_rate_from_motion(
                    sensor_data, sampling_rate
                )
                
                if hr_results:
                    results[sensor_name] = {
                        'sampling_rate': sampling_rate,
                        'hr_extraction': hr_results,
                        'raw_data': sensor_data
                    }
        
        # Compare with actual heart rate if available
        if actual_hr is not None:
            self._compare_with_actual_hr(results, actual_hr, condition)
        
        return results
    
    def _compare_with_actual_hr(self, motion_results, actual_hr_df, condition):
        """Compare extracted heart rate with actual measurements"""
        print(f"\n--- Heart Rate Comparison for {condition} ---")
        
        # Get actual heart rate statistics
        print(f"Actual HR DataFrame columns: {list(actual_hr_df.columns)}")
        print(f"Actual HR sample data:")
        print(actual_hr_df.head())
        
        # Look for heart rate values in various possible column names
        hr_cols = []
        for col in actual_hr_df.columns:
            if any(word in col.lower() for word in ['rate', 'bpm', 'heart', 'hr', 'pulse']):
                hr_cols.append(col)
        
        if not hr_cols:
            # If no obvious HR column, look at numeric columns
            numeric_cols = actual_hr_df.select_dtypes(include=[np.number]).columns
            hr_cols = [col for col in numeric_cols if col not in ['timestamp', 'time']]
        
        if hr_cols:
            hr_col = hr_cols[0]  # Use first available
            actual_hr_values = actual_hr_df[hr_col].dropna()
            
            if len(actual_hr_values) > 0:
                actual_mean = actual_hr_values.mean()
                actual_std = actual_hr_values.std()
                actual_range = (actual_hr_values.min(), actual_hr_values.max())
                
                print(f"Actual Heart Rate Stats (from {hr_col}):")
                print(f"  Mean: {actual_mean:.1f} BPM")
                print(f"  Std:  {actual_std:.1f} BPM")
                print(f"  Range: {actual_range[0]:.1f} - {actual_range[1]:.1f} BPM")
                
                # Compare with motion-extracted values
                if motion_results:
                    print(f"\nComparison with Motion-Extracted HR:")
                    
                    all_extracted = []
                    for sensor_name, sensor_results in motion_results.items():
                        hr_extraction = sensor_results['hr_extraction']
                        for axis, hr_data in hr_extraction.items():
                            estimated_bpm = hr_data['estimated_bpm']
                            confidence = hr_data['peak_ratio']
                            all_extracted.append((estimated_bpm, confidence))
                            
                            error = abs(estimated_bpm - actual_mean)
                            error_pct = (error / actual_mean) * 100
                            
                            print(f"  {sensor_name}-{axis}: {estimated_bpm:.1f} BPM "
                                  f"(error: {error:.1f} BPM, {error_pct:.1f}%, conf: {confidence:.3f})")
                    
                    if all_extracted:
                        # Find best estimate (highest confidence)
                        best_estimate, best_conf = max(all_extracted, key=lambda x: x[1])
                        best_error = abs(best_estimate - actual_mean)
                        print(f"\nBest estimate: {best_estimate:.1f} BPM "
                              f"(error: {best_error:.1f} BPM, confidence: {best_conf:.3f})")
                else:
                    print("No motion-extracted heart rate data available")
        else:
            print("Could not find heart rate column in actual data")
    
    def visualize_heart_rate_analysis(self, condition="no_caffeine"):
        """Create visualizations of heart rate extraction"""
        if condition not in self.datasets:
            return
            
        results = self.analyze_heart_rate_extraction(condition)
        if not results:
            print(f"No results to visualize for {condition}")
            return
            
        # Count total plots needed
        total_plots = 0
        for sensor_results in results.values():
            total_plots += len(sensor_results['hr_extraction'])
        
        if total_plots == 0:
            print("No heart rate extractions to visualize")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, min(3, total_plots), figsize=(15, 8))
        if total_plots == 1:
            axes = np.array([[axes, None, None], [None, None, None]])
        elif total_plots <= 3:
            axes = axes.reshape(2, -1)
        
        fig.suptitle(f'Heart Rate Extraction Analysis - {condition.title()}', fontsize=14)
        
        plot_idx = 0
        
        for sensor_name, sensor_results in results.items():
            hr_extraction = sensor_results['hr_extraction']
            
            for axis, hr_data in hr_extraction.items():
                if plot_idx >= 6:  # Limit to 6 plots
                    break
                    
                row = plot_idx // 3
                col = plot_idx % 3
                
                if axes[row, col] is not None:
                    ax = axes[row, col]
                    
                    # Plot FFT spectrum
                    freqs = hr_data['frequencies']
                    fft_vals = hr_data['fft_values']
                    
                    ax.plot(freqs * 60, fft_vals, 'b-', alpha=0.7)
                    ax.axvline(hr_data['estimated_bpm'], color='red', linestyle='--', linewidth=2,
                              label=f'Peak: {hr_data["estimated_bpm"]:.1f} BPM')
                    ax.set_xlabel('Heart Rate (BPM)')
                    ax.set_ylabel('FFT Magnitude')
                    ax.set_title(f'{sensor_name}\n{axis}', fontsize=10)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(40, 120)  # Focus on reasonable HR range
                
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 6):
            row = i // 3
            col = i % 3
            if row < 2 and col < 3 and axes[row, col] is not None:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('heart_rate.png')
        
        return results

def main():
    """Main analysis function"""
    print("Heart Rate Extraction from Tremor Data - Enhanced Version")
    print("="*55)
    
    # Initialize analyzer
    analyzer = HeartRateFFTAnalyzer()
    
    # Load data
    print("Loading sensor data...")
    analyzer.load_all_data()
    
    # Explore data structure first
    analyzer.explore_data_structure("no_caffeine")
    analyzer.explore_data_structure("caffeine")
    
    # Analyze and visualize
    print("\n" + "="*55)
    print("HEART RATE EXTRACTION ANALYSIS")
    print("="*55)
    
    # Analyze both conditions
    no_caff_results = analyzer.visualize_heart_rate_analysis("no_caffeine")
    caff_results = analyzer.visualize_heart_rate_analysis("caffeine")
    
    # Compare results if both succeeded
    if no_caff_results and caff_results:
        print(f"\n=== CAFFEINE EFFECT ON EXTRACTED HEART RATE ===")
        
        for sensor in no_caff_results:
            if sensor in caff_results:
                print(f"\n{sensor}:")
                no_caff_hr = no_caff_results[sensor]['hr_extraction']
                caff_hr = caff_results[sensor]['hr_extraction']
                
                for axis in no_caff_hr:
                    if axis in caff_hr:
                        # Use corrected BPM if available
                        before_bpm = no_caff_hr[axis].get('corrected_bpm', no_caff_hr[axis]['estimated_bpm'])
                        after_bpm = caff_hr[axis].get('corrected_bpm', caff_hr[axis]['estimated_bpm'])
                        
                        # Also show original values for comparison
                        before_orig = no_caff_hr[axis]['estimated_bpm']
                        after_orig = caff_hr[axis]['estimated_bpm']
                        
                        change = after_bpm - before_bpm
                        
                        before_corrected = no_caff_hr[axis].get('correction_applied', False)
                        after_corrected = caff_hr[axis].get('correction_applied', False)
                        
                        correction_note = ""
                        if before_corrected or after_corrected:
                            correction_note = f" (orig: {before_orig:.1f}→{after_orig:.1f})"
                        
                        print(f"  {axis}: {before_bpm:.1f} → {after_bpm:.1f} BPM ({change:+.1f}){correction_note}")
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()