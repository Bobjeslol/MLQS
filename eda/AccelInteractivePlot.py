import re
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

class AccelInteractivePlot:
    """
    Interactive accelerometer-playback widget using ipywidgets & seaborn.
    Expects each DataFrame to have columns: time, x, y, z.
    """
    def __init__(self, all_data):
        self.all_data = all_data
        # Sort dataset keys by embedded date in their names
        self.keys = sorted(all_data.keys(), key=self._extract_date)
        # Compute global time range across all datasets
        self.tmin, self.tmax = self._compute_time_range()
        # Build the ipywidget controls and output
        self._build_widgets()
    
    def _extract_date(self, key):
        """Parse YYYY-MM-DD from dataset key, fallback to datetime.min."""
        m = re.search(r'(\d{4}-\d{2}-\d{2})', key)
        return datetime.strptime(m.group(1), '%Y-%m-%d') if m else datetime.min

    def _compute_time_range(self):
        """Find the min/max time across all accelerometer DataFrames."""
        times = []
        for data_dict in self.all_data.values():
            df = data_dict.get('Accelerometer.csv')
            if df is not None:
                times.append(df['time'])
        if not times:
            raise RuntimeError("No accelerometer data found.")
        all_times = pd.concat(times)
        return float(all_times.min()), float(all_times.max())

    def _build_widgets(self):
        # One checkbox per dataset
        self.checkboxes = {
            ds: widgets.Checkbox(value=False, description=ds)
            for ds in self.keys
        }
        # Radio buttons to choose axis X, Y or Z
        self.axis_widget = widgets.RadioButtons(
            options=['X', 'Y', 'Z'],
            description='Axis:'
        )
        # Slider to select time window
        self.time_slider = widgets.FloatRangeSlider(
            value=[self.tmin, self.tmax],
            min=self.tmin,
            max=self.tmax,
            step=(self.tmax - self.tmin) / 200,
            description='Time range (s):',
            continuous_update=False,
            layout=widgets.Layout(width='90%')
        )
        # Link controls to update method
        controls = {**self.checkboxes,
                    'axis': self.axis_widget,
                    'time_range': self.time_slider}
        self.out = widgets.interactive_output(self._update, controls)

    def _update(self, **kwargs):
        """Callback: extract widget values and call plotting."""
        axis = kwargs.pop('axis')
        time_range = kwargs.pop('time_range')
        # All checked datasets
        selected = [ds for ds, val in kwargs.items() if val]
        self._plot(selected, axis, time_range)

    def _plot(self, datasets, axis, time_range):
        """Combine selected datasets and render the time-series plot."""
        plt.clf()
        if not datasets:
            plt.text(0.5, 0.5, "No datasets selected", ha='center')
            plt.axis('off')
            display(plt.gcf())
            return

        # Gather and rename columns
        records = []
        for ds in datasets:
            df = self.all_data[ds].get('Accelerometer.csv')
            if df is None:
                continue
            tmp = df[['time', 'x', 'y', 'z']].copy()
            tmp.rename(columns={
                'x': 'Acceleration_x_m/s^2',
                'y': 'Acceleration_y_m/s^2',
                'z': 'Acceleration_z_m/s^2'
            }, inplace=True)
            tmp['dataset'] = ds
            records.append(tmp)

        # Guard against no valid data
        if not records:
            plt.text(0.5, 0.5, "No accelerometer data for selected datasets", ha='center')
            plt.axis('off')
            display(plt.gcf())
            return

        combined = pd.concat(records, ignore_index=True)
        start, end = time_range
        mask = (combined['time'] >= start) & (combined['time'] <= end)
        combined = combined.loc[mask]

        # Plot with seaborn
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        col_map = {
            'X': 'Acceleration_x_m/s^2',
            'Y': 'Acceleration_y_m/s^2',
            'Z': 'Acceleration_z_m/s^2'
        }
        ycol = col_map[axis]
        sns.lineplot(
            data=combined,
            x='time', y=ycol,
            hue='dataset',
            hue_order=datasets,
            linewidth=1
        )
        plt.xlabel('Time (s)')
        plt.ylabel(f'Acceleration {axis} (m/s²)')
        plt.title(f'Combined Accel {axis} (t={start:.1f}–{end:.1f}s)')
        plt.legend(title='Dataset', loc='best')
        plt.tight_layout()
        plt.show()

    def display(self):
        """Render the full widget: checkboxes, axis selector, slider, and plot."""
        controls_box = widgets.VBox([
            widgets.Label("Select datasets (chronological):"),
            *self.checkboxes.values(),
            self.axis_widget,
            self.time_slider
        ])
        display(controls_box, self.out)
