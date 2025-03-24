import digital_rf as drf
import numpy as np
from datetime import datetime, timedelta
import zipfile
import os
from pprint import pprint

import utilrsw

def extract_zip(zip_file):
    dir_name = zip_file.replace('.zip', '')
    os.makedirs(dir_name, exist_ok=True)

    # Extract the contents of the zip file into the directory
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(dir_name)

    return dir_name

def read_one_dir(dir_name):
  do = drf.DigitalRFReader(dir_name)

  print("Reading:", dir_name)

  # Get channels
  channels = do.get_channels()
  print("Channels:\t", channels)

  # Get bounds for the first channel
  s, e = do.get_bounds(channels[0])
  print("Start Unix time:", s)
  print("End Unix time:  ", e)

  properties = do.get_properties(channels[0])
  print("Properties:")
  for key, value in properties.items():
    print(f" {key}: {value}")

  sample_rate = properties['samples_per_second']
  print("Sample rate:\t", sample_rate, "samples per second")

  print("Start UTC time: ", datetime.utcfromtimestamp(s / int(sample_rate)))
  print("End UTC time:   ", datetime.utcfromtimestamp(e / int(sample_rate)))

  # Get continuous blocks of data
  continuous_blocks = do.get_continuous_blocks(s, e, channels[0])
  print(f"\nFound {len(continuous_blocks)} continuous block(s)")

  all_data = []
  all_timestamps = []
  for block_start, block_length in continuous_blocks.items():
    data = do.read_vector(block_start, block_length, channels[0])
    all_data.append(data)

    metadata_samples = do.read_metadata(
          start_sample=s,
          end_sample=e,
          channel_name=channels[0],
      )

    # Get the start and end timestamps for the current block
    start_time = datetime.utcfromtimestamp(block_start / int(sample_rate))
    end_time = start_time + timedelta(seconds=int(block_length) / int(sample_rate))
    print("Block")
    print(" # samples:\t", len(data))
    print(" Sample rate:\t", sample_rate)
    print(" Length:\t", block_length)
    print(" Start:\t\t", start_time)
    print(" End:\t\t", end_time)
    print(" data.shape: ", data.shape)
    print(" Metadata:")
    for k, v in metadata_samples.items():
      print(f"  {k}")
      for key, value in v.items():
        print(f"   {key}: {value}")

    all_timestamps.append((start_time, end_time, sample_rate))

  all_data = np.concatenate(all_data)
  return all_timestamps, all_data

site_id = 'W2NAF'
site_id = "KD1LE node 43-3"
zip_dir = f'{site_id}/zip'
files = os.listdir(zip_dir)
files.sort()
for file in files:
  if file.endswith('.zip'):
    zip_file = os.path.join(zip_dir, file)
    dir_name = extract_zip(zip_file)
    time_, data = read_one_dir(dir_name)
    #print("Timestamps:", time_)
    #print("Data: ", data)
    #print("shape: ", data.shape)
    #exit()
