import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

class GyroInteractivePlot:
    """
    Interactive playback widget for Gyroscope data (X, Y & Z axes)
    using ipywidgets & seaborn.
    """
    def __init__(self, all_data):
        self.all_data = all_data
        # sort dataset keys by embedded date
        self.keys = sorted(all_data.keys(), key=self._extract_date)
        # compute overall time min/max from Gyroscope.csv
        self.tmin, self.tmax = self._compute_time_range()
        self._build_widgets()

    def _extract_date(self, key):
        m = re.search(r'(\d{4}-\d{2}-\d{2})', key)
        return datetime.strptime(m.group(1), '%Y-%m-%d') if m else datetime.min

    def _compute_time_range(self):
        times = []
        for data_dict in self.all_data.values():
            df = data_dict.get('Gyroscope.csv')
            if df is not None and 'Time_s' in df:
                times.append(df['Time_s'])
        if not times:
            raise RuntimeError("No time data found in any Gyroscope.csv")
        all_times = pd.concat(times)
        return float(all_times.min()), float(all_times.max())

    def _build_widgets(self):
        # axis selector
        self.axis_widget = widgets.RadioButtons(
            options=['X', 'Y', 'Z'],
            description='Axis:'
        )
        # dataset checkboxes
        self.checkboxes = {
            ds: widgets.Checkbox(value=False, description=ds)
            for ds in self.keys
        }
        # time-range slider
        self.time_slider = widgets.FloatRangeSlider(
            value=[self.tmin, self.tmax],
            min=self.tmin,
            max=self.tmax,
            step=(self.tmax - self.tmin) / 200,
            description='Time range (s):',
            continuous_update=False,
            layout=widgets.Layout(width='90%')
        )
        controls = {
            **self.checkboxes,
            'axis': self.axis_widget,
            'time_range': self.time_slider
        }
        self.out = widgets.interactive_output(self._update, controls)

    def _update(self, **kwargs):
        axis = kwargs.pop('axis')
        trange = kwargs.pop('time_range')
        selected = [ds for ds,val in kwargs.items() if val]
        self._plot(selected, axis, trange)

    def _plot(self, datasets, axis, time_range):
        plt.clf()
        if not datasets:
            plt.text(0.5, 0.5, "No datasets selected", ha='center')
            plt.axis('off')
            display(plt.gcf())
            return

        # map axis letter to column name
        col_map = {
            'X': 'Gyroscope_x_rad/s',
            'Y': 'Gyroscope_y_rad/s',
            'Z': 'Gyroscope_z_rad/s'
        }
        colname = col_map[axis]

        records = []
        for ds in datasets:
            df = self.all_data[ds].get('Gyroscope.csv')
            if df is None or colname not in df:
                continue
            tmp = df[['Time_s', colname]].copy()
            tmp.columns = ['Time_s', 'Value']
            tmp['dataset'] = ds
            records.append(tmp)

        if not records:
            plt.text(0.5, 0.5, "No gyroscope data for those selections",
                     ha='center')
            plt.axis('off')
            display(plt.gcf())
            return

        combined = pd.concat(records, ignore_index=True)
        start, end = time_range
        combined = combined[(combined['Time_s'] >= start) &
                             (combined['Time_s'] <= end)]

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=combined,
            x='Time_s', y='Value',
            hue='dataset',
            hue_order=datasets,
            linewidth=1
        )
        plt.xlabel('Time (s)')
        plt.ylabel(f"Gyroscope {axis} (rad/s)")
        plt.title(f"Gyroscope {axis} over {start:.1f}–{end:.1f}s")
        plt.legend(title='Dataset', loc='best')
        plt.tight_layout()
        plt.show()

    def display(self):
        controls_box = widgets.VBox([
            widgets.Label("Select datasets (chronological):"),
            *self.checkboxes.values(),
            self.axis_widget,
            self.time_slider
        ])
        display(controls_box, self.out)


# Usage example:
# plotter = GyroInteractivePlot(all_data)
# plotter.display()
