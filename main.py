# Agg backend for matplotlib, no display needed. It avoids memory leak issues. But slows down plotting.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
from itertools import product
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from daspy import read, DASDateTime, Collection

from core import customized_stft, sliding_window, get_encoder, FeaturesFC


def plot_raw(metadata, data_path, dest_path):
    try:
        coll = Collection(
            data_path + os.sep + "*.h5",
            timeinfo_slice = slice(31, 43),
            timeinfo_format='%Y%m%d%H%M',
            flength=60
        )
    except:
        return
    num_points = len(metadata['timestamp'])
    for i in tqdm(range(num_points), desc = "Plotting Progress:"):
        timestamp, start_channel = metadata['timestamp'].iloc[i], metadata['start_channel'].iloc[i]
        label, proba = metadata['label'].iloc[i], metadata['proba'].iloc[i]
        start_time = DASDateTime.fromtimestamp(timestamp)
        start_time = start_time.local()
        
        end_time = start_time + 60
        end_channel = start_channel + 64
        sec = coll.copy().select(start = start_time, end = end_time, 
                                 readsec = True, dtype = np.float32, 
                                 chmin = start_channel, chmax = end_channel)
        dt_str = start_time.strftime("%Y%m%d%H%M%S")
        figname = f"channel_{start_channel}_{dt_str}_label{label}_proba_{proba * 100:.2f}.png"
        sec.plot(xmode = "channel", savefig = os.path.join(dest_path, figname), dpi = 400, vmax_per = 98)
        
        if (i + 1) % 200 == 0:
            plt.close("all")

def customized_read(filepath, prev_sec):
    curr_sec = read(filepath, dtype = np.float32)
    if prev_sec is None:
        sec = curr_sec
    else:
        sec = prev_sec + curr_sec
    
    num_samples = sec.data.shape[-1]
    step_size = 80
    samples_used = (num_samples // step_size) * step_size - step_size
    prev_sec = sec.copy().trimming(spmin = samples_used)
    return sec, prev_sec

def clustering(metadata):
    # DBSCAN
    result = pd.DataFrame(columns = metadata.columns)
    metadata['normed_timestamp'] = metadata['timestamp'] - metadata['timestamp'].min()
    metadata['normed_timestamp'] = metadata['normed_timestamp'] / 52
    metadata['normed_channel'] = metadata['start_channel'] / 64

    dbscan = DBSCAN(eps = 1.5, min_samples = 2)
    data = metadata[['normed_timestamp', 'normed_channel']].copy()
    cluster_labels = dbscan.fit_predict(data)
    metadata['cluster'] = cluster_labels
    grouped = metadata.groupby('cluster')
    for cluster_id, group in grouped:
        if cluster_id == -1:
            continue
        start_channel = (group['start_channel'] * group['proba']).sum() / group['proba'].sum()
        proba = group['proba'].mean()
        timestamp = group['timestamp'].min()
        label = round(group['label'].mean())
        result.loc[len(result)] = [timestamp, int(start_channel), int(label), round(proba, 4)]
    return result

if __name__ == "__main__":
    ### INPUT AREA ###
    # latent_dim select which encoder to use 16, 32, 64, 128
    latent_dim = 16
    ### INPUT PATH and OUTPUT PATH ###
    data_path = r"F:\feixi\Beijing\processed\202512081059"
    subfolder_name = os.path.basename(data_path)
    dest_parent_path = r"C:\Users\Yi\Desktop\classification_result"

    PLOT = True
    ### END OF INPUT AREA ###

    encoder = get_encoder(latent_dim)
    classifier = FeaturesFC(latent_dim, num_classes=4)
    checkpoint = torch.load(os.path.join("models", f"Classifier_FC_{latent_dim}.pth"), weights_only = True)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to("cuda")
    classifier.eval()

    prev_sec = None
    dest_path = os.path.join(dest_parent_path, subfolder_name)
    os.makedirs(dest_path, exist_ok=True)
    csv_name = subfolder_name + ".csv"

    metadata = pd.DataFrame(columns = ["timestamp", "start_channel", "label", "proba"])

    for filename in tqdm(os.listdir(data_path), desc = "Traversing Files:"):
        if not filename.endswith(".h5"):
            continue

        sec, prev_sec = customized_read(os.path.join(data_path, filename), prev_sec)

        start_times, start_channels, stft_splits = customized_stft(sec, noverlap=0)
        dataset = torch.tensor(stft_splits, device = "cuda")
        with torch.no_grad():
            encoded = encoder(dataset)
        start_time_ids, splits = sliding_window(encoded.cpu().numpy(), 0, 64, 32)
        sub_start_times = start_times[start_time_ids]
        temp_data = splits.swapaxes(1, 2)
        temp_data = torch.tensor(temp_data, device = "cuda")
        with torch.no_grad():
            proba = classifier(temp_data)
        proba = proba.to("cpu")
        if len(proba.shape) == 2:
            if len(start_time_ids) == 1:
                proba = proba.unsqueeze(0)
            elif len(start_channels) == 1:
                proba = proba.unsqueeze(1)
            else:
                raise IndexError
        labels = torch.argmax(proba, dim = -1).numpy()
        proba = torch.nn.functional.softmax(proba, dim = -1).numpy()

        for time_id, channel_id in product(range(len(sub_start_times)), range(len(start_channels))):
            label = labels[time_id][channel_id]
            if label == 0:
                continue
            start_time = sub_start_times[time_id]
            timestamp = sec.start_time + start_time
            timestamp = timestamp.timestamp()
            start_channel = start_channels[channel_id]
            probability = proba[time_id][channel_id][label]
            
            metadata.loc[len(metadata)] = [timestamp, int(start_channel), int(label), round(probability, 4)]

    metadata = metadata.astype({
        'start_channel': 'int32',
        'label': 'int32',
        'timestamp': 'float64',
        'proba': 'float64'
    })
    print(f"Number of events before clustering: {len(metadata)}! Clustering the events...")
    metadata = clustering(metadata)
    metadata = metadata.astype({
        'start_channel': 'int32',
        'label': 'int32',
        'timestamp': 'float64',
        'proba': 'float64'
    })
    metadata.to_csv(os.path.join(dest_path, csv_name), index=False)
    if PLOT:
        print(f"Number of events after clustering: {len(metadata)}! Plotting...")
        plot_raw(metadata, data_path, dest_path)
    