import numpy as np
import pandas as pd

def p90(x):
    """Return the 90th percentile of a sequence."""
    return np.percentile(x, 90)

class FeatureExtractor:
    """
    A class to load sensor data from a Parquet file and
    bin/aggregate it by a specified time interval, with optional filtering.
    """
    def __init__(self, file_path: str, sensor_aggs: dict = None):
        """
        Initialize the FeatureExtractor with the path to a Parquet file and
        an optional sensor aggregation mapping.

        Parameters:
        -----------
        file_path : str
            Path to the Parquet file containing the raw sensor data.
        sensor_aggs : dict, optional
            Mapping of sensor column names to a list of aggregation functions or names.
            e.g. {
                'accelerometer_x': ['mean', 'max', 'std', p90],
                'gyroscope_y': ['mean', 'max', 'std'],
                'linear_acceleration_z': ['mean', 'max']
            }
            If None, defaults to accelerometer axes with mean, max, std, p90.
        """
        self.file_path = file_path
        self.df = pd.read_parquet(file_path)

        # Default sensor aggregations
        if sensor_aggs is None:
            self.sensor_aggs = {
                'accelerometer_x': ['mean', 'max', 'std'],
                'accelerometer_y': ['mean', 'max', 'std'],
                'accelerometer_z': ['mean', 'max', 'std']
            }
        else:
            self.sensor_aggs = sensor_aggs

    def bin_data(self,
                 bin_size: float = 0.025,
                 experiment_id=None,
                 base=None,
                 caffeine_ml=None,
                 source=None,
                 save: bool = False,
                 save_path: str = None
                ) -> pd.DataFrame:
        """
        Bin and aggregate the sensor data into fixed time intervals,
        optionally filtering by metadata.

        Parameters:
        -----------
        bin_size : float
            Width of each time bin in seconds (e.g., 0.025 for 25 ms).
        experiment_id, base, caffeine_ml, source : optional
            Filter values for each metadata column. If provided, only rows
            matching the value are retained.
        save : bool
            Whether to save the resulting DataFrame to disk.
        save_path : str, optional
            File path to save the binned DataFrame. If None and save=True,
            defaults to 'grouped_bin_{bin_size*1000:.0f}ms.parquet'.

        Returns:
        --------
        pd.DataFrame
            A DataFrame grouped by ['experiment_id', 'base', 'caffeine_ml', 'source', 't_bin']
            with aggregated columns for every sensor defined in sensor_aggs.
        """
        # Copy and optionally filter metadata
        df = self.df.copy()
        if experiment_id is not None:
            df = df[df['experiment_id'] == experiment_id]
        if base is not None:
            df = df[df['base'] == base]
        if caffeine_ml is not None:
            df = df[df['caffeine_ml'] == caffeine_ml]
        if source is not None:
            df = df[df['source'] == source]

        # Create the time-bin column
        df['t_bin'] = np.ceil(df['time'] / bin_size) * bin_size

        # Use all sensors specified in sensor_aggs
        agg_funcs = {col: funcs for col, funcs in self.sensor_aggs.items()}

        # Group and aggregate
        grouped = (
            df
            .groupby(['experiment_id', 'base', 'caffeine_ml', 'source', 't_bin'])
            .agg(agg_funcs)
            .reset_index()
        )

        # Flatten MultiIndex columns
        grouped.columns = [
            f"{col[0]}_{col[1] if isinstance(col, tuple) else ''}".rstrip('_')
            for col in grouped.columns
        ]

        # Save if requested
        if save:
            if save_path is None:
                ms = int(bin_size * 1000)
                save_path = f"grouped_bin_{ms}ms.parquet"
            grouped.to_parquet(save_path, index=False)

        return grouped

# Example usage:
# extractor = FeatureExtractor('output-all.parquet')
# df25 = extractor.bin_data(
#     bin_size=0.025,
#     experiment_id='2025-06-11',
#     caffeine_ml=230,
#     source='Timo',
#     save=True
# )
# print(df25.head())
