import os
import zipfile

import numpy as np

import digital_rf as drf
from datetime import datetime, timedelta

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

  print("\nReading:", dir_name)

  # Get channels
  channels = do.get_channels()
  print("  Channels:\t  ", channels)

  # Get bounds for the first channel
  start, end = do.get_bounds(channels[0])
  print("  Channel start Unix time:", start)
  print("  Channel stop Unix time: ", end)

  properties = do.get_properties(channels[0])
  print("  Properties:")
  for key, value in properties.items():
    print(f"    {key}: {value}")

  sample_rate = properties['samples_per_second']
  print("  Computed start UTC time: ", datetime.utcfromtimestamp(start / int(sample_rate)))
  print("  Computed stop UTC time:  ", datetime.utcfromtimestamp(end / int(sample_rate)))

  # Get continuous blocks of data
  continuous_blocks = do.get_continuous_blocks(start, end, channels[0])
  plural = "s" if len(continuous_blocks) > 1 else ""
  print(f"\n  Found {len(continuous_blocks)} continuous block{plural}")

  metadata_samples = do.read_metadata(
        start_sample=start,
        end_sample=end,
        channel_name=channels[0],
    )

  all_data = []
  all_timestamps = []
  bn = 0
  for block_start, block_length in continuous_blocks.items():
    bn = bn + 1
    data = do.read_vector(block_start, block_length, channels[0])
    all_data.append(data)


    # Get the start and end timestamps for the current block.
    # TODO: Need to use epoch as it may not always be Unix epoch as assumed below.
    start_time = datetime.utcfromtimestamp(block_start / int(sample_rate))
    end_time = start_time + timedelta(seconds=(int(block_length)-1) / int(sample_rate))
    print("    Block", bn)
    print("      # samples:      ", len(data))
    print("      data.shape:     ", data.shape)
    print("      Sample rate:    ", sample_rate)
    print("      Block start:    ", block_start)
    print("      Block length:   ", block_length)
    print("      Computed start: ", start_time)
    print("      Computed stop:  ", end_time)
    print("      Metadata:")
    for key, value in metadata_samples[block_start].items():
      print(f"        {key}: {value}")

    all_timestamps.append((start_time, end_time, sample_rate))

  all_data = np.concatenate(all_data)
  return all_timestamps, all_data

import sys
zip_dir = sys.argv[1]
files = os.listdir(zip_dir)
files.sort()
for file in files:
  if file.endswith('.zip'):
    zip_file = os.path.join(zip_dir, file)
    dir_name = extract_zip(zip_file)
    time_, data = read_one_dir(dir_name)
