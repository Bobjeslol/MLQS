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
    def __init__(self, data_path="../data/Timo/", zip_files=None):
        self.data_path = data_path
        # zip_files: dict mapping session_name -> zip filename
        self.zip_files = zip_files or {
            'base_2025-06-06': 'base 2025-06-06 15-45-11.zip',
            'base_2025-06-07': 'base 2025-06-07 12-17-44.zip', 
            '200ml_2025-06-07': '200 ml 2025-06-07 12-52-34.zip',
            '230ml_2025-06-06': '230ml 2025-06-06 16-22-45.zip'
        }
        self.baseline_data = {}
        self.caffeine_data = {}
        self.all_data = {}

    def extract_and_load_data(self):
        """Extract zip files and load sensor data"""
        # Use self.zip_files for session/zip mapping
        for session_name, zip_filename in self.zip_files.items():
            zip_path = os.path.join(self.data_path, zip_filename)
            if os.path.exists(zip_path):
                print(f"\nLoading {session_name} from {zip_filename}")
                session_data = self._extract_and_parse_zip(zip_path)
                self.all_data[session_name] = session_data
                
                # Categorize data
                if 'base' in session_name:
                    self.baseline_data[session_name] = session_data
                else:
                    self.caffeine_data[session_name] = session_data
            else:
                print(f"Warning: {zip_path} not found")

        print(f"\nData loading summary:")
        print(f"Baseline sessions: {len(self.baseline_data)}")
        print(f"Caffeine sessions: {len(self.caffeine_data)}")
        print(f"Total sessions: {len(self.all_data)}")

    def _extract_and_parse_zip(self, zip_path):
        """Extract zip and parse all sensor files"""
        data_dict = {}

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"  Files in {os.path.basename(zip_path)}:")
                for file in file_list:
                    print(f"    - {file}")

                for file_name in file_list:
                    # Focus on the three main sensor CSV files, skip meta folder
                    if (file_name.endswith('.csv') and 
                        any(sensor in file_name for sensor in ['Gyroscope.csv', 'Accelerometer.csv', 'Linear Acceleration.csv']) and
                        'meta/' not in file_name):
                        
                        try:
                            with zip_ref.open(file_name) as file:
                                content = file.read().decode('utf-8')
                                
                                # Parse as CSV with header row
                                df = pd.read_csv(pd.io.common.StringIO(content), header=0)
                                
                                # Clean column names
                                df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '') 
                                             for col in df.columns]
                                
                                # Convert all columns to numeric where possible
                                for col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors='ignore')
                                
                                # Remove any completely empty rows
                                df = df.dropna(how='all')
                                
                                data_dict[file_name] = df
                                print(f"      Loaded {file_name} as CSV: shape {df.shape}")
                                
                        except Exception as e:
                            print(f"      Error loading {file_name}: {e}")
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is not a valid zip file")
        except FileNotFoundError:
            print(f"Error: {zip_path} not found")

        return data_dict

    def explore_data_structure(self):
        """Explore and print data structure"""
        print("\n" + "="*60)
        print("DATA STRUCTURE ANALYSIS")
        print("="*60)

        for session_name, dataset in self.all_data.items():
            print(f"\n--- {session_name.upper()} ---")
            self._analyze_dataset(dataset, session_name)

    def _analyze_dataset(self, dataset, name):
        """Analyze individual dataset structure"""
        if not dataset:
            print(f"  No data found for {name}")
            return

        for file_name, data in dataset.items():
            print(f"\nFile: {file_name}")

            if isinstance(data, pd.DataFrame):
                print(f"  Type: DataFrame")
                print(f"  Shape: {data.shape}")
                print(f"  Columns: {list(data.columns)}")
                
                # Check for common phyphox column patterns
                acc_cols = [col for col in data.columns if 'acc' in col.lower()]
                gyro_cols = [col for col in data.columns if 'gyro' in col.lower()]
                linear_acc_cols = [col for col in data.columns if 'linear' in col.lower()]
                time_cols = [col for col in data.columns if 'time' in col.lower() or 't_s' in col.lower()]
                
                print(f"  Accelerometer columns: {acc_cols}")
                print(f"  Gyroscope columns: {gyro_cols}")
                print(f"  Linear acceleration columns: {linear_acc_cols}")
                print(f"  Time columns: {time_cols}")

                # Show sample data
                print("  Sample data (first 3 rows):")
                print(data.head(3).to_string(index=False))

                # Basic statistics for numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    print(f"  Numeric columns summary:")
                    print(data[numeric_cols].describe().round(6))

            else:
                print(f"  Type: {type(data)}")
                if isinstance(data, str):
                    lines = data.split('\n')[:5]  # First 5 lines
                    print(f"  Content preview (first 5 lines):")
                    for i, line in enumerate(lines):
                        print(f"    {i+1}: {line}")

    def calculate_tremor_metrics(self, df, sensor_type="auto"):
        """Calculate tremor-related metrics from sensor data"""
        metrics = {}
        acc_cols = [col for col in df.columns if 'acc' in col.lower() and 'linear' not in col.lower()]
        linear_acc_cols = [col for col in df.columns if 'linear' in col.lower()]
        gyro_cols = [col for col in df.columns if 'gyro' in col.lower()]
        time_cols = [col for col in df.columns if 'time' in col.lower() or 't_s' in col.lower()] # Might be useful in feature engineering
        sensor_cols = acc_cols + linear_acc_cols + gyro_cols

        if not sensor_cols:
            print("    No sensor columns found")
            return metrics

        print(f"    Analyzing columns: {sensor_cols}")

        for col in sensor_cols:
            if col in df.columns:
                signal = df[col].dropna()
                if len(signal) > 10:  # Need reasonable amount of data
                    # Basic statistical metrics
                    metrics[f"{col}_mean"] = signal.mean()
                    metrics[f"{col}_std"] = signal.std()
                    metrics[f"{col}_rms"] = np.sqrt(np.mean(signal**2))
                    metrics[f"{col}_range"] = signal.max() - signal.min()
                    
                    # Tremor-specific metrics
                    # Root mean square of differences (measure of variability)
                    if len(signal) > 1:
                        diff_signal = np.diff(signal)
                        metrics[f"{col}_diff_rms"] = np.sqrt(np.mean(diff_signal**2))
                        
                        # High frequency content approximation
                        metrics[f"{col}_high_freq_energy"] = np.sum(diff_signal**2)
                    
                    # Coefficient of variation (normalized variability)
                    if abs(signal.mean()) > 1e-10:
                        metrics[f"{col}_cv"] = signal.std() / abs(signal.mean())
                    
                    # Percentile-based metrics (robust to outliers)
                    metrics[f"{col}_iqr"] = signal.quantile(0.75) - signal.quantile(0.25)
                    metrics[f"{col}_mad"] = (signal - signal.mean()).abs().mean()  # Mean absolute deviation

        return metrics

    def compare_conditions(self):
        """Compare tremor metrics between baseline and caffeine conditions"""
        print("\n" + "="*60)
        print("TREMOR COMPARISON ANALYSIS")
        print("="*60)

        if not self.baseline_data or not self.caffeine_data:
            print("Need both baseline and caffeine data for comparison")
            return {}

        results = {}
        
        # Get all baseline files
        baseline_files = set()
        for session_data in self.baseline_data.values():
            baseline_files.update(session_data.keys())
            
        # Get all caffeine files  
        caffeine_files = set()
        for session_data in self.caffeine_data.values():
            caffeine_files.update(session_data.keys())

        print(f"Baseline files: {baseline_files}")
        print(f"Caffeine files: {caffeine_files}")

        # Find common file types (may have different exact names but same sensor type)
        common_files = baseline_files.intersection(caffeine_files)
        
        if not common_files:
            print("No exactly matching files found. Analyzing all sensor files separately...")
            # Analyze each file type separately
            all_baseline_metrics = {}
            all_caffeine_metrics = {}
            
            # Process baseline data
            for session_name, session_data in self.baseline_data.items():
                print(f"\nProcessing baseline session: {session_name}")
                for file_name, df in session_data.items():
                    if isinstance(df, pd.DataFrame):
                        metrics = self.calculate_tremor_metrics(df)
                        all_baseline_metrics[f"{session_name}_{file_name}"] = metrics
                        
            # Process caffeine data
            for session_name, session_data in self.caffeine_data.items():
                print(f"\nProcessing caffeine session: {session_name}")
                for file_name, df in session_data.items():
                    if isinstance(df, pd.DataFrame):
                        metrics = self.calculate_tremor_metrics(df)
                        all_caffeine_metrics[f"{session_name}_{file_name}"] = metrics
            
            # Aggregate metrics across sessions
            results = self._aggregate_and_compare_metrics(all_baseline_metrics, all_caffeine_metrics)
        else:
            # Process matching files
            for file_name in common_files:
                print(f"\nAnalyzing: {file_name}")
                results[file_name] = self._compare_matching_files(file_name)

        return results

    def _aggregate_and_compare_metrics(self, baseline_metrics, caffeine_metrics):
        """Aggregate metrics across sessions and compare"""
        # Collect all unique metric names
        all_metric_names = set()
        for metrics in baseline_metrics.values():
            all_metric_names.update(metrics.keys())
        for metrics in caffeine_metrics.values():
            all_metric_names.update(metrics.keys())
            
        aggregated_comparison = {}
        
        for metric_name in all_metric_names:
            # Collect values for this metric from baseline sessions
            baseline_values = []
            for session_metrics in baseline_metrics.values():
                if metric_name in session_metrics:
                    baseline_values.append(session_metrics[metric_name])
                    
            # Collect values for this metric from caffeine sessions  
            caffeine_values = []
            for session_metrics in caffeine_metrics.values():
                if metric_name in session_metrics:
                    caffeine_values.append(session_metrics[metric_name])
            
            if baseline_values and caffeine_values:
                baseline_mean = np.mean(baseline_values)
                caffeine_mean = np.mean(caffeine_values)
                
                # Calculate percentage change
                if abs(baseline_mean) > 1e-10:
                    pct_change = ((caffeine_mean - baseline_mean) / abs(baseline_mean)) * 100
                else:
                    pct_change = 0 if caffeine_mean == 0 else float('inf')
                
                aggregated_comparison[metric_name] = {
                    'baseline_mean': baseline_mean,
                    'caffeine_mean': caffeine_mean,
                    'baseline_values': baseline_values,
                    'caffeine_values': caffeine_values,
                    'difference': caffeine_mean - baseline_mean,
                    'percent_change': pct_change
                }
                
                print(f"  {metric_name}:")
                print(f"    Baseline (n={len(baseline_values)}): {baseline_mean:.6f}")
                print(f"    Caffeine (n={len(caffeine_values)}): {caffeine_mean:.6f}")
                print(f"    Change: {pct_change:+.2f}%")
        
        return {'aggregated': aggregated_comparison}

    def _compare_matching_files(self, file_name):
        """Compare metrics for a specific file across baseline and caffeine sessions"""
        baseline_metrics_list = []
        caffeine_metrics_list = []
        
        # Collect baseline metrics for this file
        for _, session_data in self.baseline_data.items():
            if file_name in session_data:
                df = session_data[file_name]
                if isinstance(df, pd.DataFrame):
                    metrics = self.calculate_tremor_metrics(df)
                    baseline_metrics_list.append(metrics)
        
        # Collect caffeine metrics for this file
        for _, session_data in self.caffeine_data.items():
            if file_name in session_data:
                df = session_data[file_name]
                if isinstance(df, pd.DataFrame):
                    metrics = self.calculate_tremor_metrics(df)
                    caffeine_metrics_list.append(metrics)
        
        if not baseline_metrics_list or not caffeine_metrics_list:
            print(f"Insufficient data for {file_name}, measurement should be rejected")
            exit(0)
        
        # Average metrics across sessions
        all_metric_names = set()
        for metrics in baseline_metrics_list + caffeine_metrics_list:
            all_metric_names.update(metrics.keys())
        
        comparison = {}
        for metric_name in all_metric_names:
            # Get baseline values
            baseline_values = [m[metric_name] for m in baseline_metrics_list if metric_name in m]
            caffeine_values = [m[metric_name] for m in caffeine_metrics_list if metric_name in m]
            
            if baseline_values and caffeine_values:
                baseline_mean = np.mean(baseline_values)
                caffeine_mean = np.mean(caffeine_values)
                
                # Calculate percentage change
                if abs(baseline_mean) > 1e-10:
                    pct_change = ((caffeine_mean - baseline_mean) / abs(baseline_mean)) * 100
                else:
                    pct_change = 0 if caffeine_mean == 0 else float('inf')
                
                comparison[metric_name] = {
                    'no_caffeine': baseline_mean,  # Keep original naming for compatibility
                    'caffeine': caffeine_mean,
                    'difference': caffeine_mean - baseline_mean,
                    'percent_change': pct_change
                }
                
                print(f"    {metric_name}:")
                print(f"      Baseline: {baseline_mean:.6f}")
                print(f"      Caffeine: {caffeine_mean:.6f}")
                print(f"      Change: {pct_change:+.2f}%")
        
        return comparison

    def visualize_results(self, results):
        """Create visualizations of the comparison results"""
        if not results:
            print("No results to visualize")
            return None

        # Handle aggregated results
        if 'aggregated' in results:
            comparison_data = results['aggregated']
        else:
            # Handle individual file results
            comparison_data = {}
            for file_results in results.values():
                comparison_data.update(file_results)

        if not comparison_data:
            print("No comparison data to visualize")
            return None

        # Prepare data for plotting
        all_metrics = []
        for metric, values in comparison_data.items():
            all_metrics.append({
                'metric': metric,
                'baseline': values['baseline_mean'] if 'baseline_mean' in values else values['no_caffeine'],
                'caffeine': values['caffeine_mean'] if 'caffeine_mean' in values else values['caffeine'],
                'percent_change': values['percent_change']
            })

        df_metrics = pd.DataFrame(all_metrics)

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Caffeine Effect on Hand Tremor Metrics\n(Baseline vs Caffeinated Conditions)', fontsize=16)

        # Plot 1: Distribution of percent changes
        ax1 = axes[0, 0]
        df_metrics.boxplot(column='percent_change', ax=ax1)
        ax1.set_title('Distribution of Percent Changes')
        ax1.set_ylabel('Percent Change (%)')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No change')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Before vs After scatter
        ax2 = axes[0, 1]
        ax2.scatter(df_metrics['baseline'], df_metrics['caffeine'], alpha=0.6, s=50)
        ax2.set_xlabel('Baseline Values')
        ax2.set_ylabel('Caffeine Values')
        ax2.set_title('Baseline vs Caffeine Comparison')

        # Add diagonal line (no change line)
        min_val = min(df_metrics['baseline'].min(), df_metrics['caffeine'].min())
        max_val = max(df_metrics['baseline'].max(), df_metrics['caffeine'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='No change')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Largest changes (both increases and decreases)
        ax3 = axes[1, 0]
        # Get top 10 absolute changes
        df_sorted = df_metrics.reindex(df_metrics['percent_change'].abs().sort_values(ascending=False).index)
        top_changes = df_sorted.head(10)
        
        colors = ['red' if x > 0 else 'blue' for x in top_changes['percent_change']]
        bars = ax3.barh(range(len(top_changes)), top_changes['percent_change'], color=colors, alpha=0.7)
        ax3.set_yticks(range(len(top_changes)))
        ax3.set_yticklabels([f"{row['metric'][:25]}..." if len(row['metric']) > 25 else row['metric']
                           for _, row in top_changes.iterrows()], fontsize=8)
        ax3.set_xlabel('Percent Change (%)')
        ax3.set_title('Largest Changes (Red=Increase, Blue=Decrease)')
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)

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

        ax4.axis('off')
        summary_text = []
        for key, value in summary_stats.items():
            if 'Metrics' in key or 'Total' in key:
                summary_text.append(f"{key}: {int(value)}")
            else:
                summary_text.append(f"{key}: {value:.2f}")
        
        ax4.text(0.1, 0.9, '\n'.join(summary_text), transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax4.set_title('Summary Statistics')

        plt.tight_layout()
        plt.savefig('caffeine_tremor_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as 'caffeine_tremor_analysis.png'")
        plt.show()

        return df_metrics

    def generate_summary_report(self, results, df_metrics=None):
        """Reports on findings"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)

        if df_metrics is not None and len(df_metrics) > 0:
            mean_change = df_metrics['percent_change'].mean()
            median_change = df_metrics['percent_change'].median()
            std_change = df_metrics['percent_change'].std()
            increased_count = (df_metrics['percent_change'] > 0).sum()
            decreased_count = (df_metrics['percent_change'] < 0).sum()
            total_count = len(df_metrics)

            print("QUANTITATIVE RESULTS:")
            print(f"   Total metrics analyzed: {total_count}")
            print(f"   Metrics increased: {increased_count} ({increased_count/total_count*100:.1f}%)")
            print(f"   Metrics decreased: {decreased_count} ({decreased_count/total_count*100:.1f}%)")
            print(f"   Average change: {mean_change:+.2f}%")
            print(f"   Median change: {median_change:+.2f}%")
            print(f"   Standard deviation: {std_change:.2f}%")

            # Identify most significant changes
            top_increases = df_metrics.nlargest(3, 'percent_change')
            top_decreases = df_metrics.nsmallest(3, 'percent_change')

            print("TOP INCREASES:")
            for _, row in top_increases.iterrows():
                print(f"   {row['metric']}: {row['percent_change']:+.1f}%")

            print("TOP DECREASES:")
            for _, row in top_decreases.iterrows():
                print(f"   {row['metric']}: {row['percent_change']:+.1f}%")

            print("INTERPRETATION:")
            if abs(mean_change) > 10:
                print(f"STRONG SIGNAL: Average change of {mean_change:+.1f}% suggests caffeine has a detectable effect on tremor metrics.")
            elif abs(mean_change) > 5:
                print(f"MODERATE SIGNAL: Average change of {mean_change:+.1f}% suggests potential caffeine effect, but may need more data.")
            else:
                print(f"WEAK SIGNAL: Average change of {mean_change:+.1f}% suggests minimal effect or need for improved methodology.")

            # Consistency check
            if std_change > abs(mean_change) * 2:
                print(f"High variability ({std_change:.1f}%) suggests heterogeneous effects across different metrics or measurement conditions.")
            else:
                print(f"Relatively consistent effects across metrics.")

def main():
    """Main analysis function"""
    print("Timo's data tremor analysis")
    print("="*40)

    analyzer = CaffeineAnalyzer()

    print("Loading data...")
    analyzer.extract_and_load_data()
    print("\nExploring data structure...")
    analyzer.explore_data_structure()
    print("\nComparing conditions...")
    results = analyzer.compare_conditions()
    if results:
        print(f"\nCreating visualizations...")
        df_metrics = analyzer.visualize_results(results)
        analyzer.generate_summary_report(results, df_metrics)
    else:
        print("Results empty.")
        exit(1)

if __name__ == "__main__":
    main()
