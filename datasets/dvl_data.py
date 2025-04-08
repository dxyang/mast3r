import csv
import numpy as np
import pandas as pd

class DvlDataset:
    """Dataset class that loads DVL data from CSV file """

    def __init__(
        self,
        csv_file: str,
    ):
        self.csv_file = csv_file
        self.csv_df = pd.read_csv(self.csv_file)

        assert 'ros_time' in self.csv_df.columns, "CSV file must contain 'ros_time' column"
        assert 'avg_beam_range' in self.csv_df.columns, "CSV file must contain 'avg_beam_range' column"

        self.timestamps = (self.csv_df['ros_time'].values * 1e9).astype(np.uint64)  # Convert to nanoseconds
        self.distances = self.csv_df['avg_beam_range'].values
        self.beam1 = self.csv_df['dvl_beam_1_range'].values
        self.beam2 = self.csv_df['dvl_beam_2_range'].values
        self.beam3 = self.csv_df['dvl_beam_3_range'].values
        self.beam4 = self.csv_df['dvl_beam_4_range'].values

        self.range_data = np.stack([self.beam1, self.beam2, self.beam3, self.beam4], axis=1)
        self.avg_range = np.mean(self.range_data, axis=1)
        self.std_range = np.std(self.range_data, axis=1)

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, item: int):
        return self.csv_df.iloc[item]

    def get_range_at_timestamp(self, ros_time: int):
        if ros_time <= self.timestamps[0]:
            idx = 0  # Return the first distance if timestamp is before the range
        elif ros_time >= self.timestamps[-1]:
            idx = -1  # Return the last distance if timestamp is after the range
        else:
            # Find indices of the two nearest timestamps
            idx = np.searchsorted(self.timestamps, ros_time)

        if idx == 0 or idx == -1:
            interpolated_beam1 = self.beam1[idx]
            interpolated_beam2 = self.beam2[idx]
            interpolated_beam3 = self.beam3[idx]
            interpolated_beam4 = self.beam4[idx]
        else:
            t1, t2 = self.timestamps[idx - 1], self.timestamps[idx]

            beam1_1, beam1_2 = self.beam1[idx - 1], self.beam1[idx]
            beam2_1, beam2_2 = self.beam2[idx - 1], self.beam2[idx]
            beam3_1, beam3_2 = self.beam3[idx - 1], self.beam3[idx]
            beam4_1, beam4_2 = self.beam4[idx - 1], self.beam4[idx]

            # Interpolate each beam
            interpolated_beam1 = beam1_1 + (beam1_2 - beam1_1) * (ros_time - t1) / (t2 - t1)
            interpolated_beam2 = beam2_1 + (beam2_2 - beam2_1) * (ros_time - t1) / (t2 - t1)
            interpolated_beam3 = beam3_1 + (beam3_2 - beam3_1) * (ros_time - t1) / (t2 - t1)
            interpolated_beam4 = beam4_1 + (beam4_2 - beam4_1) * (ros_time - t1) / (t2 - t1)

        # calculate meand and std dev
        avg_range = np.mean([interpolated_beam1, interpolated_beam2, interpolated_beam3, interpolated_beam4])
        std_range = np.std([interpolated_beam1, interpolated_beam2, interpolated_beam3, interpolated_beam4])

        return avg_range, std_range


if __name__ == "__main__":
    csv_fp = "/home/dayang/code/mast3r/datasets/dvl_data.csv"
    dataset = DvlDataset(csv_file=csv_fp)

    for idx in range(len(dataset)):
        dataset[idx]

    print(dataset.get_range_at_timestamp(563229901049))