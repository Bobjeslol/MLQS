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
    """
    def __init__(self, all_data):
        self.all_data = all_data
        self.keys = sorted(all_data.keys(), key=self._extract_date)
        self.tmin, self.tmax = self._compute_time_range()
        self._build_widgets()
    
    def _extract_date(self, key):
        m = re.search(r'(\d{4}-\d{2}-\d{2})', key)
        return datetime.strptime(m.group(1), '%Y-%m-%d') if m else datetime.min

    def _compute_time_range(self):
        times = []
        for data_dict in self.all_data.values():
            df = data_dict.get('Accelerometer.csv')
            if df is not None:
                times.append(df['Time_s'])
        if not times:
            raise RuntimeError("No accelerometer data found.")
        all_times = pd.concat(times)
        return float(all_times.min()), float(all_times.max())

    def _build_widgets(self):
        # Dataset checkboxes
        self.checkboxes = {
            ds: widgets.Checkbox(value=False, description=ds)
            for ds in self.keys
        }
        # Axis selector
        self.axis_widget = widgets.RadioButtons(
            options=['X', 'Y'],
            description='Axis:'
        )
        # Time-range slider
        self.time_slider = widgets.FloatRangeSlider(
            value=[self.tmin, self.tmax],
            min=self.tmin,
            max=self.tmax,
            step=(self.tmax - self.tmin) / 200,
            description='Time range (s):',
            continuous_update=False,
            layout=widgets.Layout(width='90%')
        )
        # Wire up interactive output
        controls = {**self.checkboxes,
                    'axis': self.axis_widget,
                    'time_range': self.time_slider}
        self.out = widgets.interactive_output(self._update, controls)

    def _update(self, **kwargs):
        axis = kwargs.pop('axis')
        time_range = kwargs.pop('time_range')
        selected = [ds for ds, val in kwargs.items() if val]
        self._plot(selected, axis, time_range)

    def _plot(self, datasets, axis, time_range):
        plt.clf()
        if not datasets:
            plt.text(0.5, 0.5, "No datasets selected", ha='center')
            plt.axis('off')
            display(plt.gcf())
            return

        # Combine and filter data
        records = []
        for ds in datasets:
            df = self.all_data[ds].get('Accelerometer.csv')
            if df is None:
                continue
            tmp = df[['Time_s', 'Acceleration_x_m/s^2', 'Acceleration_y_m/s^2']].copy()
            tmp['dataset'] = ds
            records.append(tmp)
        combined = pd.concat(records, ignore_index=True)
        start, end = time_range
        mask = (combined['Time_s'] >= start) & (combined['Time_s'] <= end)
        combined = combined.loc[mask]

        # Plot
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        ycol = 'Acceleration_x_m/s^2' if axis == 'X' else 'Acceleration_y_m/s^2'
        sns.lineplot(
            data=combined,
            x='Time_s', y=ycol,
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
        """Render the full widget (checkboxes + slider + plot)."""
        controls_box = widgets.VBox([
            widgets.Label("Select datasets (chronological):"),
            *self.checkboxes.values(),
            self.axis_widget,
            self.time_slider
        ])
        display(controls_box, self.out)
