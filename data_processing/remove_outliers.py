import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class SensorOutlierDetector:
    def __init__(self, k_neighbors=20, outlier_threshold_std=2):
        """
        Initialize the outlier detector.
        
        Args:
            k_neighbors: Number of neighbors for LOF calculation (default: 20)
            outlier_threshold_std: Standard deviations above mean for outlier threshold (default: 2)
        """
        self.k_neighbors = k_neighbors
        self.outlier_threshold_std = outlier_threshold_std
        self.sensor_types = ['Accelerometer.csv', 'Gyroscope.csv', 'Linear Acceleration.csv']
        
    def load_sensor_data(self, zip_path):
        """
        Load sensor data from a zip file.
        
        Args:
            zip_path: Path to the zip file containing sensor data
            
        Returns:
            dict: Dictionary with sensor type as key and DataFrame as value
        """
        data = {}
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for sensor_type in self.sensor_types:
                try:
                    with zip_ref.open(sensor_type) as file:
                        df = pd.read_csv(file)
                        # Assuming columns are: time, x, y, z
                        df.columns = ['time', 'x', 'y', 'z']
                        data[sensor_type.replace('.csv', '')] = df
                        print(f"Loaded {sensor_type}: {len(df)} data points")
                except KeyError:
                    print(f"Warning: {sensor_type} not found in {zip_path}")
                    
        return data
    
    def calculate_lof_scores(self, xyz_data):
        """
        Calculate LOF scores for 3D sensor data.
        
        Args:
            xyz_data: numpy array of shape (n_samples, 3) containing x, y, z coordinates
            
        Returns:
            tuple: (lof_scores, outlier_mask, threshold)
        """
        # Initialize LOF detector
        lof = LocalOutlierFactor(n_neighbors=self.k_neighbors, contamination='auto')
        
        # Fit and predict (-1 for outliers, 1 for inliers)
        outlier_labels = lof.fit_predict(xyz_data)
        
        # Get the negative LOF scores (more negative = more outlier-like)
        lof_scores = -lof.negative_outlier_factor_
        
        # Calculate threshold based on mean + std_multiplier * std
        lof_mean = np.mean(lof_scores)
        lof_std = np.std(lof_scores)
        threshold = lof_mean + self.outlier_threshold_std * lof_std
        
        # Create outlier mask based on our threshold
        outlier_mask = lof_scores > threshold
        
        return lof_scores, outlier_mask, threshold
    
    def detect_outliers(self, sensor_data):
        """
        Detect outliers for each sensor type in the data.
        
        Args:
            sensor_data: Dictionary with sensor data DataFrames
            
        Returns:
            dict: Dictionary with outlier information for each sensor
        """
        results = {}
        
        for sensor_type, df in sensor_data.items():
            print(f"\nProcessing {sensor_type}...")
            
            # Extract x, y, z coordinates
            xyz_data = df[['x', 'y', 'z']].values
            
            # Calculate LOF scores and detect outliers
            lof_scores, outlier_mask, threshold = self.calculate_lof_scores(xyz_data)
            
            # Store results
            results[sensor_type] = {
                'original_data': df.copy(),
                'lof_scores': lof_scores,
                'outlier_mask': outlier_mask,
                'threshold': threshold,
                'n_outliers': np.sum(outlier_mask),
                'outlier_percentage': (np.sum(outlier_mask) / len(df)) * 100
            }
            
            print(f"  Total points: {len(df)}")
            print(f"  Outliers detected: {np.sum(outlier_mask)} ({results[sensor_type]['outlier_percentage']:.2f}%)")
            print(f"  LOF threshold: {threshold:.4f}")
            print(f"  LOF score range: {np.min(lof_scores):.4f} - {np.max(lof_scores):.4f}")
            
        return results
    
    def remove_outliers(self, results):
        """
        Remove outliers from the original data.
        
        Args:
            results: Dictionary from detect_outliers method
            
        Returns:
            dict: Dictionary with cleaned data for each sensor
        """
        cleaned_data = {}
        
        for sensor_type, result in results.items():
            # Keep only non-outlier points
            clean_mask = ~result['outlier_mask']
            cleaned_df = result['original_data'][clean_mask].copy()
            cleaned_data[sensor_type] = cleaned_df
            
            print(f"{sensor_type}: {len(result['original_data'])} -> {len(cleaned_df)} points")
            
        return cleaned_data
    
    def plot_outlier_analysis(self, results, sensor_type, output_folder, filename):
        """
        Create visualization plots for outlier analysis.
        
        Args:
            results: Dictionary from detect_outliers method
            sensor_type: Which sensor to plot
        """
        if sensor_type not in results:
            print(f"Sensor type {sensor_type} not found in results")
            return
            
        result = results[sensor_type]
        df = result['original_data']
        lof_scores = result['lof_scores']
        outlier_mask = result['outlier_mask']
        threshold = result['threshold']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'LOF Outlier Analysis - {sensor_type}', fontsize=16)
        
        # 1. 3D scatter plot
        ax1 = plt.subplot(2, 2, 1, projection='3d')
        inliers = ~outlier_mask
        
        ax1.scatter(df.loc[inliers, 'x'], df.loc[inliers, 'y'], df.loc[inliers, 'z'], 
                   c='blue', alpha=0.6, s=20, label='Inliers')
        ax1.scatter(df.loc[outlier_mask, 'x'], df.loc[outlier_mask, 'y'], df.loc[outlier_mask, 'z'], 
                   c='red', alpha=0.8, s=40, label='Outliers')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Scatter Plot')
        ax1.legend()
        
        # 2. LOF scores histogram
        ax2 = plt.subplot(2, 2, 2)
        ax2.hist(lof_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
        ax2.set_xlabel('LOF Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('LOF Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Time series with outliers marked
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(df['time'], np.sqrt(df['x']**2 + df['y']**2 + df['z']**2), 
                color='blue', alpha=0.6, label='Magnitude')
        outlier_times = df.loc[outlier_mask, 'time']
        outlier_magnitudes = np.sqrt(df.loc[outlier_mask, 'x']**2 + 
                                   df.loc[outlier_mask, 'y']**2 + 
                                   df.loc[outlier_mask, 'z']**2)
        ax3.scatter(outlier_times, outlier_magnitudes, color='red', s=40, alpha=0.8, label='Outliers')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Magnitude')
        ax3.set_title('Time Series with Outliers')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. LOF scores over time
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(df['time'], lof_scores, color='green', alpha=0.7)
        ax4.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
        ax4.scatter(df.loc[outlier_mask, 'time'], lof_scores[outlier_mask], 
                   color='red', s=40, alpha=0.8, label='Outliers')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('LOF Score')
        ax4.set_title('LOF Scores Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_folder}//{filename}_{sensor_type}.png")
    
    def analyze_variance(self, original_data, cleaned_data):
        """
        Analyze variance before and after outlier removal.
        
        Args:
            original_data: Dictionary with original sensor data
            cleaned_data: Dictionary with cleaned sensor data
        """
        print("\nVariance Analysis (Before -> After outlier removal):")
        print("="*60)
        
        for sensor_type in original_data.keys():
            if sensor_type in cleaned_data:
                orig_df = original_data[sensor_type]
                clean_df = cleaned_data[sensor_type]
                
                print(f"\n{sensor_type}:")
                for axis in ['x', 'y', 'z']:
                    orig_var = orig_df[axis].var()
                    clean_var = clean_df[axis].var()
                    reduction = ((orig_var - clean_var) / orig_var) * 100
                    print(f"  {axis}-axis variance: {orig_var:.6f} -> {clean_var:.6f} ({reduction:.1f}% reduction)")

def process_zip_file(zip_path, output_folder, filename):
    """
    Process a single zip file for outlier detection.
    
    Args:
        zip_path: Path to the zip file
    """
    detector = SensorOutlierDetector(k_neighbors=20, outlier_threshold_std=2)
    
    # Load data
    print(f"Processing: {zip_path}")
    sensor_data = detector.load_sensor_data(zip_path)
    
    if not sensor_data:
        print("No sensor data found!")
        return None
    
    # Detect outliers
    results = detector.detect_outliers(sensor_data)
    
    # Remove outliers
    cleaned_data = detector.remove_outliers(results)
    
    # Analyze variance
    detector.analyze_variance(sensor_data, cleaned_data)
    
    # Plot analysis for each sensor type
    for sensor_type in results.keys():
        detector.plot_outlier_analysis(results, sensor_type, output_folder, filename)
    
    return {
        'original_data': sensor_data,
        'cleaned_data': cleaned_data,
        'outlier_results': results
    }

# Process all files in /data_trimmed/Timo
if __name__ == "__main__":
   zip_folder = "../data_trimmed/Timo/"
   output_folder = "../data_cleaned/Timo/"
   
   # Create output directory if it doesn't exist
   os.makedirs(output_folder, exist_ok=True)
   
   zip_files = [f for f in os.listdir(zip_folder) if f.endswith('.zip')]
   
   all_results = {}
   for zip_file in zip_files:
       zip_path = os.path.join(zip_folder, zip_file)
       output_zip_path = os.path.join(output_folder, zip_file)
       
       print(f"\n{'='*50}")
       print(f"Processing: {zip_file}")
       
       # Process the zip file
       result = process_zip_file(zip_path, output_folder, zip_file)
       if result is None:
           continue
           
       all_results[zip_file] = result
       cleaned_data = result['cleaned_data']
       
       # Create new zip file with cleaned data
       with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as new_zip:
           # Copy meta folder from original zip
           with zipfile.ZipFile(zip_path, 'r') as orig_zip:
               for file_info in orig_zip.infolist():
                   if file_info.filename.startswith('meta/'):
                       new_zip.writestr(file_info, orig_zip.read(file_info.filename))
           
           # Write cleaned CSV files
           for sensor_type, cleaned_df in cleaned_data.items():
               csv_filename = f"{sensor_type}.csv"
               csv_content = cleaned_df.to_csv(index=False)
               new_zip.writestr(csv_filename, csv_content)
       
       print(f"Created cleaned zip: {output_zip_path}")