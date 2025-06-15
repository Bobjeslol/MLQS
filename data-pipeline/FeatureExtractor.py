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
            If None, defaults to accelerometer, gyroscope, and linear acceleration axes
            with mean, max, std, and p90.
        """
        self.file_path = file_path
        self.df = pd.read_parquet(file_path)

        # Default sensor aggregations covering all three sensor types
        if sensor_aggs is None:
            aggs = ['mean', 'max', 'std']
            self.sensor_aggs = {
                'accelerometer_x': aggs,
                'accelerometer_y': aggs,
                'accelerometer_z': aggs,
                'gyroscope_x':    aggs,
                'gyroscope_y':    aggs,
                'gyroscope_z':    aggs,
                'linear_acceleration_x': aggs,
                'linear_acceleration_y': aggs,
                'linear_acceleration_z': aggs
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

        Returns:
        --------
        pd.DataFrame
            A DataFrame grouped by ['experiment_id', 'base', 'caffeine_ml', 'source', 't_bin']
            with aggregated columns for every sensor defined in sensor_aggs.
        """
        df = self.df.copy()
        # apply any filters provided
        if experiment_id is not None:
            df = df[df['experiment_id'] == experiment_id]
        if base is not None:
            df = df[df['base'] == base]
        if caffeine_ml is not None:
            df = df[df['caffeine_ml'] == caffeine_ml]
        if source is not None:
            df = df[df['source'] == source]

        # compute time bins
        df['t_bin'] = np.ceil(df['time'] / bin_size) * bin_size

        # group and aggregate based on sensor_aggs
        grouped = (
            df
            .groupby(['experiment_id', 'base', 'caffeine_ml', 'source', 't_bin'])
            .agg(self.sensor_aggs)
            .reset_index()
        )

        # flatten MultiIndex columns
        grouped.columns = [
            f"{col[0]}_{col[1] if isinstance(col, tuple) else ''}".rstrip('_')
            for col in grouped.columns
        ]

        # save if requested
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
