import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import butter, filtfilt
import zipfile
import os
import json
from pathlib import Path

class CaffeineAnalyzer:
    def __init__(self, data_path="data/Nathaniel/"):
        self.data_path = data_path
        self.no_caffeine_data = None
        self.caffeine_data = None
        
    def extract_and_load_data(self):
        """Extract zip files and load sensor data"""
        no_caff_zip = os.path.join(self.data_path, "No_caffeine-2025-06-04_13-45-27.zip")
        caff_zip = os.path.join(self.data_path, "30min_20mg_caffeine-2025-06-04_14-17-49.zip")
        
        # Extract and load no caffeine data
        self.no_caffeine_data = self._extract_and_parse_zip(no_caff_zip)
        print(f"No caffeine data loaded: {len(self.no_caffeine_data)} files")
        
        # Extract and load caffeine data  
        self.caffeine_data = self._extract_and_parse_zip(caff_zip)
        print(f"Caffeine data loaded: {len(self.caffeine_data)} files")
        
    def _extract_and_parse_zip(self, zip_path):
        """Extract zip and parse all sensor files"""
        data_dict = {}
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"\nFiles in {os.path.basename(zip_path)}:")
            for file in file_list:
                print(f"  - {file}")
            
            for file_name in file_list:
                if file_name.endswith('.csv') or file_name.endswith('.txt') or file_name.endswith('.json'):
                    try:
                        with zip_ref.open(file_name) as file:
                            content = file.read().decode('utf-8')
                            
                            # Try to parse as CSV first
                            try:
                                df = pd.read_csv(pd.io.common.StringIO(content))
                                data_dict[file_name] = df
                                print(f"    Loaded {file_name} as CSV: shape {df.shape}")
                            except:
                                # If CSV fails, try JSON
                                try:
                                    json_data = json.loads(content)
                                    data_dict[file_name] = json_data
                                    print(f"    Loaded {file_name} as JSON")
                                except:
                                    # Store as raw text
                                    data_dict[file_name] = content
                                    print(f"    Loaded {file_name} as raw text")
                    except Exception as e:
                        print(f"    Error loading {file_name}: {e}")
        
        return data_dict
    
    def explore_data_structure(self):
        """Explore and print data structure"""
        print("\n" + "="*50)
        print("DATA STRUCTURE ANALYSIS")
        print("="*50)
        
        print("\n--- NO CAFFEINE DATA ---")
        self._analyze_dataset(self.no_caffeine_data, "No Caffeine")
        
        print("\n--- CAFFEINE DATA ---")
        self._analyze_dataset(self.caffeine_data, "Caffeine")
        
    def _analyze_dataset(self, dataset, name):
        """Analyze individual dataset structure"""
        for file_name, data in dataset.items():
            print(f"\nFile: {file_name}")
            
            if isinstance(data, pd.DataFrame):
                print(f"  Type: DataFrame")
                print(f"  Shape: {data.shape}")
                print(f"  Columns: {list(data.columns)}")
                print(f"  Data types: {data.dtypes.to_dict()}")
                
                # Show sample data
                print("  Sample data (first 3 rows):")
                print(data.head(3).to_string(index=False))
                
                # Basic statistics for numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    print(f"  Numeric columns stats:")
                    print(data[numeric_cols].describe())
                    
            elif isinstance(data, dict):
                print(f"  Type: Dictionary")
                print(f"  Keys: {list(data.keys())}")
            else:
                print(f"  Type: {type(data)}")
                print(f"  Content preview: {str(data)[:200]}...")
    
    def calculate_tremor_metrics(self, df, sensor_type="accelerometer"):
        """Calculate tremor-related metrics from sensor data"""
        metrics = {}
        
        # Identify relevant columns based on sensor type
        if sensor_type == "accelerometer":
            cols = [col for col in df.columns if any(axis in col.lower() for axis in ['x', 'y', 'z', 'acc'])]
        elif sensor_type == "gyroscope":
            cols = [col for col in df.columns if any(axis in col.lower() for axis in ['gyro', 'angular'])]
        else:
            # Use all numeric columns
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not cols:
            return metrics
            
        print(f"Analyzing {sensor_type} columns: {cols}")
        
        for col in cols:
            if col in df.columns:
                signal = df[col].dropna()
                if len(signal) > 0:
                    # Basic statistics
                    metrics[f"{col}_mean"] = signal.mean()
                    metrics[f"{col}_std"] = signal.std()
                    metrics[f"{col}_rms"] = np.sqrt(np.mean(signal**2))
                    
                    # Tremor-specific metrics
                    # High-frequency content (potential tremor band 4-12 Hz)
                    if len(signal) > 10:  # Need enough samples
                        # Simple high-pass filter approximation
                        diff_signal = np.diff(signal)
                        metrics[f"{col}_high_freq_energy"] = np.sum(diff_signal**2)
                        
                        # Peak-to-peak amplitude
                        metrics[f"{col}_peak_to_peak"] = signal.max() - signal.min()
                        
                        # Coefficient of variation (std/mean ratio)
                        if abs(signal.mean()) > 1e-10:
                            metrics[f"{col}_cv"] = signal.std() / abs(signal.mean())
        
        return metrics
    
    def compare_conditions(self):
        """Compare tremor metrics between no caffeine and caffeine conditions"""
        print("\n" + "="*50)
        print("TREMOR COMPARISON ANALYSIS")
        print("="*50)
        
        # Find matching sensor files
        no_caff_files = set(self.no_caffeine_data.keys())
        caff_files = set(self.caffeine_data.keys())
        
        # Look for accelerometer and gyroscope data
        sensor_types = ['accelerometer', 'gyro', 'acc', 'sensor']
        
        results = {}
        
        for file_name in no_caff_files.intersection(caff_files):
            if any(sensor in file_name.lower() for sensor in sensor_types):
                print(f"\nAnalyzing: {file_name}")
                
                no_caff_df = self.no_caffeine_data[file_name]
                caff_df = self.caffeine_data[file_name]
                
                if isinstance(no_caff_df, pd.DataFrame) and isinstance(caff_df, pd.DataFrame):
                    # Calculate metrics for both conditions
                    no_caff_metrics = self.calculate_tremor_metrics(no_caff_df)
                    caff_metrics = self.calculate_tremor_metrics(caff_df)
                    
                    # Compare metrics
                    comparison = self._compare_metrics(no_caff_metrics, caff_metrics, file_name)
                    results[file_name] = comparison
        
        return results
    
    def _compare_metrics(self, no_caff_metrics, caff_metrics, file_name):
        """Compare metrics between conditions and perform statistical tests"""
        comparison = {}
        
        common_metrics = set(no_caff_metrics.keys()).intersection(set(caff_metrics.keys()))
        
        print(f"\n  Comparing {len(common_metrics)} metrics:")
        
        for metric in common_metrics:
            no_caff_val = no_caff_metrics[metric]
            caff_val = caff_metrics[metric]
            
            # Calculate percentage change
            if abs(no_caff_val) > 1e-10:
                pct_change = ((caff_val - no_caff_val) / abs(no_caff_val)) * 100
            else:
                pct_change = 0 if caff_val == 0 else float('inf')
            
            comparison[metric] = {
                'no_caffeine': no_caff_val,
                'caffeine': caff_val,
                'difference': caff_val - no_caff_val,
                'percent_change': pct_change
            }
            
            print(f"    {metric}:")
            print(f"      No caffeine: {no_caff_val:.6f}")
            print(f"      Caffeine:    {caff_val:.6f}")
            print(f"      Change:      {pct_change:+.2f}%")
        
        return comparison
    
    def visualize_results(self, results):
        """Create visualizations of the comparison results"""
        if not results:
            print("No results to visualize")
            return
        
        # Prepare data for plotting
        all_metrics = []
        
        for file_name, comparison in results.items():
            for metric, values in comparison.items():
                all_metrics.append({
                    'file': file_name,
                    'metric': metric,
                    'no_caffeine': values['no_caffeine'],
                    'caffeine': values['caffeine'],
                    'percent_change': values['percent_change']
                })
        
        if not all_metrics:
            print("No metrics to plot")
            return
            
        df_metrics = pd.DataFrame(all_metrics)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Caffeine Effect on Hand Tremor Metrics', fontsize=16)
        
        # Plot 1: Percent changes
        ax1 = axes[0, 0]
        df_metrics.boxplot(column='percent_change', ax=ax1)
        ax1.set_title('Distribution of Percent Changes')
        ax1.set_ylabel('Percent Change (%)')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Plot 2: Before vs After scatter
        ax2 = axes[0, 1]
        ax2.scatter(df_metrics['no_caffeine'], df_metrics['caffeine'], alpha=0.6)
        ax2.set_xlabel('No Caffeine')
        ax2.set_ylabel('Caffeine')
        ax2.set_title('Before vs After Caffeine')
        
        # Add diagonal line
        min_val = min(df_metrics['no_caffeine'].min(), df_metrics['caffeine'].min())
        max_val = max(df_metrics['no_caffeine'].max(), df_metrics['caffeine'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        # Plot 3: Top changes
        ax3 = axes[1, 0]
        top_changes = df_metrics.nlargest(10, 'percent_change')
        ax3.barh(range(len(top_changes)), top_changes['percent_change'])
        ax3.set_yticks(range(len(top_changes)))
        ax3.set_yticklabels([f"{row['metric'][:20]}..." if len(row['metric']) > 20 else row['metric'] 
                           for _, row in top_changes.iterrows()], fontsize=8)
        ax3.set_xlabel('Percent Change (%)')
        ax3.set_title('Largest Increases')
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        summary_stats = {
            'Mean % Change': df_metrics['percent_change'].mean(),
            'Median % Change': df_metrics['percent_change'].median(),
            'Std % Change': df_metrics['percent_change'].std(),
            'Metrics Increased': (df_metrics['percent_change'] > 0).sum(),
            'Metrics Decreased': (df_metrics['percent_change'] < 0).sum(),
            'Total Metrics': len(df_metrics)
        }
        
        y_pos = range(len(summary_stats))
        ax4.barh(y_pos, list(summary_stats.values()))
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(list(summary_stats.keys()))
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig('initial_analysis.png')
        
        return df_metrics

def main():
    """Main analysis function"""
    print("Caffeine Tremor Analysis")
    print("="*30)
    
    # Initialize analyzer
    analyzer = CaffeineAnalyzer()
    
    # Load data
    print("Loading data...")
    analyzer.extract_and_load_data()
    
    # Explore structure
    analyzer.explore_data_structure()
    
    # Compare conditions
    results = analyzer.compare_conditions()
    
    # Visualize results
    if results:
        print(f"\nFound {len(results)} sensor files to compare")
        df_metrics = analyzer.visualize_results(results)
        
        # Summary conclusion
        print("\n" + "="*50)
        print("CONCLUSION")
        print("="*50)
        
        if df_metrics is not None and len(df_metrics) > 0:
            mean_change = df_metrics['percent_change'].mean()
            increased_count = (df_metrics['percent_change'] > 0).sum()
            total_count = len(df_metrics)
            
            print(f"Total metrics analyzed: {total_count}")
            print(f"Metrics that increased: {increased_count} ({increased_count/total_count*100:.1f}%)")
            print(f"Average percent change: {mean_change:+.2f}%")
            
            if abs(mean_change) > 5:
                print(f"✓ Detectable difference found! Average change of {mean_change:+.1f}%")
            else:
                print(f"? Small difference detected. May need more data or refined methodology.")
        else:
            print("No metrics could be calculated. Check data structure.")
    else:
        print("No comparable sensor files found between conditions.")

if __name__ == "__main__":
    main()
