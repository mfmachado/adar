import numpy as np
import pandas as pd
import urllib.request
from datetime import datetime
import mne

def to_timestamp(x: str, acq_time: datetime):
    date = datetime.strptime(x, '%H:%M:%S')
    date = datetime(acq_time.year, acq_time.month, acq_time.day, date.hour, date.minute, date.second)
    if date.hour < 12:
        date = datetime(date.year, date.month, date.day + 1, date.hour, date.minute, date.second)
    return (date.day - acq_time.day) * 24 * 3600 + (date.hour - acq_time.hour) * 3600 + (
                date.minute - acq_time.minute) * 60 + date.second - acq_time.second


def subsample(data: np.ndarray, fs: int, fs_subsample: int, axis=0):
    assert fs % fs_subsample == 0
    factor = int(fs / fs_subsample)
    if data.shape[axis] % factor != 0:
        print('Subsampling led to loss of %i samples, in an online setting consider using a BlockBuffer with a '
              'buffer size of a multiple of %i samples.' % (data.shape[axis] % factor, factor))
    idx_mask = np.arange(data.shape[axis], step=factor)
    return data.take(idx_mask, axis)


def read_annotation_file(path_filename: str, acq_time: datetime):
    df_annotations_data = pd.read_csv(path_filename, sep='\t', skiprows=20)
    df_annotations_data = df_annotations_data.loc[df_annotations_data["Event"].str.startswith("SLEEP")]
    df_annotations_data = df_annotations_data.rename(columns={"Time [hh:mm:ss]": "onset", "Duration[s]": "duration"})
    df_annotations_data["timestamp"] = df_annotations_data.onset.apply(lambda x: to_timestamp(x, acq_time))

    return df_annotations_data


def merge_data_file_and_annotations(data, df_annotations, fs_new):
    raw_data = data.get_data()
    fs = int(data.info["sfreq"])
    df = pd.DataFrame(raw_data.T, columns=data.info.ch_names)
    df["times"] = data.times
    df = pd.DataFrame(data=subsample(df.to_numpy(), fs, fs_new), columns=df.columns)

    labels = [np.nan] * df.shape[0]
    for i in range(df_annotations.shape[0] - 1):
        row_start = df_annotations.iloc[i]
        row_end = df_annotations.iloc[i + 1]
        start_pos = row_start.timestamp * fs_new
        end_pos = row_end.timestamp * fs_new
        if start_pos < len(labels):
            labels[start_pos:end_pos] = [row_start["Sleep Stage"]] * (end_pos - start_pos)

    df["sleepstage"] = labels
    df = df.dropna()
    return df


if __name__ == '__main__':
    """
    This script downloads the subject data (the edf and txt file) from the CAP Sleep Database.
    It merges the txt and EDF data into a pandas DataFrame, optionally if fs_new is not None a downsampling procedure is 
    performed.
    
    """
    # Subject ID
    subject_name = 'nfle1'
    # New frequency
    fs_new = 128
    # CSV Filename
    new_filename = f'{subject_name}_data_and_annotations.csv'

    data_filename = f'{subject_name}.edf'
    filename_annotations = f'{subject_name}_annotations.csv'
    url_data = f'https://physionet.org/files/capslpdb/1.0.0/{subject_name}.edf?download'
    url_annotations = f'https://physionet.org/files/capslpdb/1.0.0/{subject_name}.txt?download'

    print(f'Downloading data file: {url_data}')
    #urllib.request.urlretrieve(url_data, data_filename)

    print(f'Downloading annotations data file: {url_annotations}')
    urllib.request.urlretrieve(url_annotations, filename_annotations)

    print(f'Loading File: {data_filename}')
    data_polysomnography = mne.io.read_raw_edf(data_filename)

    print(f'Loading Annotations file: {filename_annotations}')
    df_annotations_polysomnography = read_annotation_file(filename_annotations, data_polysomnography.info["meas_date"])

    print("\n\n Example Annotations file: \n")
    df_annotations_polysomnography.head(4)

    print(f'Merge files annotation and polysomnography data. Downsampling data from {data_polysomnography.info["sfreq"]}'
          f'to {fs_new} Hz')
    df_data = merge_data_file_and_annotations(data_polysomnography, df_annotations_polysomnography, fs_new)
    df_data.to_csv(new_filename, index=False)

    print(f'Created CSV File: {new_filename}')
    df_data.head(4)
