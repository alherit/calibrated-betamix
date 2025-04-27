# adapted from https://github.com/freelunchtheorem/Conditional_Density_Estimation
# MIT License

# Copyright (c) 2017 Jonas Rothfuss

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from urllib.request import urlretrieve
import os
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = None 

UCI_DATASETS = []

def _UCI(C):
    UCI_DATASETS.append(C)
    return C

class Dataset:

    ndim_x = None
    ndim_y = None

    target_columns = []
    feature_columns = []

    data_file_name = ''
    download_url = ''

    @property
    def data_file_path(self):
        return os.path.join(DATA_DIR, self.data_file_name)

    @property
    def needs_download(self):
        return not os.path.isfile(self.data_file_path)

    def download_dataset(self):
        print("Downloading data file from %s" % self.download_url)
        urlretrieve(self.download_url, self.data_file_path)

    def get_df(self):
        if False: #self.needs_download:
            self.download_dataset()
        df = pd.read_csv(self.data_file_path)
        return self._process_df(df)

    def get_target_feature_split(self):
        df = self.get_df()
        X = np.array(df[self.feature_columns])
        Y = np.array(df[self.target_columns])
        assert X.ndim == Y.ndim == 2
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == self.ndim_x and Y.shape[1] == self.ndim_y
        return X, Y

    def get_train_valid_splits(self, valid_portion, n_splits, shuffle=False, random_state=None):

        X, Y = self.get_target_feature_split()

        n_instances = X.shape[0]
        valid_size = int(valid_portion * n_instances)
        assert valid_size * n_splits <= n_instances

        idx = np.arange(n_instances)
        if shuffle:
            if random_state is not None:
                random_state.shuffle(idx)
            else:
                np.random.shuffle(idx)

        X_trains, Y_trains, X_valids, Y_valids = [], [], [], []

        for i in reversed(range(n_splits)):
            idx_start = (n_instances // n_splits) * i
            idx_end = idx_start + valid_size

            idx_train, idx_valid = np.concatenate([idx[:idx_start], idx[idx_end:]]),  idx[idx_start:idx_end]

            assert len(set(idx_train) | set(idx_valid)) == n_instances

            X_trains.append(X[idx_train, :])
            Y_trains.append(Y[idx_train, :])
            X_valids.append(X[idx_valid, :])
            Y_valids.append(Y[idx_valid, :])

        return X_trains, Y_trains, X_valids, Y_valids

    def _process_df(self, df):
        return df

    def __str__(self):
        return "%s (ndim_x = %i, ndim_y = %i)"%(str(self.__class__.__name__), self.ndim_x, self.ndim_y)



class NCYTaxiDropoffPredict(Dataset):

    ndim_x = 6
    ndim_y = 2

    data_file_name = 'yellow_tripdata_2016-01.csv'
    data_file_name_processed = 'yellow_tripdata_2016-01_processed.csv'
    download_url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-01.csv'

    target_columns = ['dropoff_loc_lat', 'dropoff_loc_lon']
    feature_columns = ['pickup_loc_lat', 'pickup_loc_lon', 'pickup_time_day_of_week_sin', 'pickup_time_day_of_week_cos',
                       'pickup_time_of_day_sin', 'pickup_time_of_day_cos']

    x_bounds = [-74.04, -73.75]
    y_bounds = [40.62, 40.86]
    too_close_radius = 0.00001
    min_duration = 30
    max_duration = 3 * 3600

    def __init__(self, data_folder, n_samples=10**4, seed=22):
        global DATA_DIR
        self.data_folder = data_folder
        self.n_samples = n_samples
        self.random_state = np.random.RandomState(seed)
        DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), data_folder))


    def get_df(self):
        try:
            from azureml.fsspec import AzureMachineLearningFileSystem

            #from azure.ai.ml import MLClient
            #from azure.identity import DefaultAzureCredential
            #ml_client = MLClient.from_config(credential=DefaultAzureCredential())
            #data_asset = ml_client.data.get("UCI_datasets", version=1)
            #fs = AzureMachineLearningFileSystem(data_asset.path)
            fs = AzureMachineLearningFileSystem(self.data_folder)

            print("Using Azure Filesystem, with data_folder: ", self.data_folder)
            open_file = fs.open
            data_file_path_processed = os.path.join(self.data_folder, self.data_file_name_processed)

        except:
            open_file = open

            #data_file_path_processed = os.path.join(self.data_folder, self.data_file_name_processed)
            data_file_path_processed = os.path.join(DATA_DIR, self.data_file_name_processed)
            if not os.path.isfile(data_file_path_processed):
                df = super(NCYTaxiDropoffPredict, self).get_df().dropna()
                print("save processed NYC data as csv to %s" % data_file_path_processed)
                df.to_csv(data_file_path_processed)            


        print("opening %s" % data_file_path_processed)
        with open_file(data_file_path_processed) as f:
            df = pd.read_csv(f)

        return df.sample(n=self.n_samples, random_state=self.random_state)

    def _process_df(self, df): # does some data cleaning
        data = df.values

        pickup_loc = np.array((data[:, 5], data[:, 6])).T
        dropoff_loc = np.array((data[:, 9], data[:, 10])).T

        ind = np.ones(len(data)).astype(bool)
        ind[data[:, 5] < self.x_bounds[0]] = False
        ind[data[:, 5] > self.x_bounds[1]] = False
        ind[data[:, 6] < self.y_bounds[0]] = False
        ind[data[:, 6] > self.y_bounds[1]] = False

        ind[data[:, 9] < self.x_bounds[0]] = False
        ind[data[:, 9] > self.x_bounds[1]] = False
        ind[data[:, 10] < self.y_bounds[0]] = False
        ind[data[:, 10] > self.y_bounds[1]] = False

        print('discarding {} out of bounds {} {}'.format(np.sum(np.invert(ind).astype(int)), self.x_bounds, self.y_bounds))

        early_stop = ((data[:, 5] - data[:, 9]) ** 2 + (data[:, 6] - data[:, 10]) ** 2 < self.too_close_radius)
        ind[early_stop] = False
        print('discarding {} trip less than {} gp dist'.format(np.sum(early_stop.astype(int)), self.too_close_radius ** 0.5))

        times = np.array([_process_time(d_pickup, d_dropoff) for (d_pickup, d_dropoff) in data[:, 1:3]])
        pickup_time = times[:, :2]
        dropoff_time = times[:, 2:4]
        duration = times[:, 4]

        short_journeys = (duration < self.min_duration)
        ind[short_journeys] = False
        print('discarding {} less than {}s journeys'.format(np.sum(short_journeys.astype(int)), self.min_duration))

        long_journeys = (duration > self.max_duration)
        ind[long_journeys] = False
        print('discarding {} more than {}h journeys'.format(np.sum(long_journeys.astype(int)), self.max_duration / 3600.))

        pickup_loc_lat = pickup_loc[ind, 0]
        pickup_loc_lon = pickup_loc[ind, 1]

        dropoff_loc_lat = dropoff_loc[ind, 0]
        dropoff_loc_lon = dropoff_loc[ind, 1]

        pickup_time_day_of_week = pickup_time[ind, 0]
        pickup_time_of_day = pickup_time[ind, 1]

        dropoff_time_day_of_week = dropoff_time[ind, 0]
        dropoff_time_of_day = dropoff_time[ind, 1]

        duration = duration[ind]

        print('{} total rejected journeys'.format(np.sum(np.invert(ind).astype(int))))

        df_processed = pd.DataFrame(
            {"pickup_loc_lat": pickup_loc_lat,
             "pickup_loc_lon": pickup_loc_lon,
             "dropoff_loc_lat": dropoff_loc_lat,
             "dropoff_loc_lon": dropoff_loc_lon,
             "pickup_time_day_of_week": pickup_time_day_of_week.astype(int),
             "pickup_time_day_of_week_sin": np.sin(pickup_time_day_of_week),
             "pickup_time_day_of_week_cos": np.cos(pickup_time_day_of_week.astype(int)),
             "pickup_time_of_day": pickup_time_of_day,
             "pickup_time_of_day_sin": np.sin(pickup_time_of_day),
             "pickup_time_of_day_cos": np.cos(pickup_time_of_day),
             "dropoff_time_day_of_week": dropoff_time_day_of_week.astype(int),
             "dropoff_time_of_day": dropoff_time_of_day, "duration": duration})

        return df_processed

    def __str__(self):
        return "%s (n_samples = %i, ndim_x = %i, ndim_y = %i)" % (str(self.__class__.__name__), self.n_samples, self.ndim_x, self.ndim_y)



""" helper methods """

def _convert_to_day_minute(d):
  rescale = lambda x, a, b: b[0] + (b[1] - b[0]) * x / (a[1] - a[0])

  day_of_week = rescale(float(d.weekday()), [0, 6], [0, 2 * np.pi])
  time_of_day = rescale(d.time().hour * 60 + d.time().minute, [0, 24 * 60], [0, 2 * np.pi])
  return day_of_week, time_of_day


def _process_time(pickup_datetime, dropoff_datetime):
  d_pickup = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
  d_dropoff = datetime.strptime(dropoff_datetime, "%Y-%m-%d %H:%M:%S")
  duration = (d_dropoff - d_pickup).total_seconds()

  pickup_day_of_week, pickup_time_of_day = _convert_to_day_minute(d_pickup)
  dropoff_day_of_week, dropoff_time_of_day = _convert_to_day_minute(d_dropoff)

  return [pickup_day_of_week, pickup_time_of_day, dropoff_day_of_week, dropoff_time_of_day, duration]

