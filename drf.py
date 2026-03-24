import os

import utilrsw

format = u"%(message)s"
log = utilrsw.logger("drf", log_dir="log", console_format=format, file_format=format)


def info(msg):
  log.info(msg)


def error(msg):
  log.error(f"Error: {msg}")


def cli():
  import argparse

  parser = argparse.ArgumentParser(
      description='Read Digital RF data from PSWS station directories.'
  )
  parser.add_argument(
      '--base_dir', required=True,
      help='Directory containing station subdirectories (e.g., S0000183, N000020).'
  )
  parser.add_argument(
      '--station', default=None,
      help='Only process this station subdirectory.'
  )
  parser.add_argument(
      '--n', type=int, default=1,
      help='Max number of OBS subdirectories to process per station. Defaults to 1; -1 means process all OBS subdirectories.'
  )
  parser.add_argument(
      '--read_data', action='store_true',
      help='Whether to read data blocks. If not set, only metadata is read and data blocks are not read.'
  )
  parser.add_argument(
      '--return_data', action='store_true',
      help='Whether to return data blocks. If not set, data blocks are read but not concatenated and returned to caller.'
  )

  args = parser.parse_args()

  if args.n == -1:
    args.n = None

  if args.return_data:
    args.read_data = True

  return args


def dirs(base_dir):
  dir_list = sorted(os.listdir(base_dir))
  # Keep only directories
  return [d for d in dir_list if os.path.isdir(os.path.join(base_dir, d))]


def process_observation(observation_dir):
  import numpy as np
  from datetime import datetime, timedelta

  import digital_rf as drf

  do = drf.DigitalRFReader(observation_dir)

  info(f"Reading: {observation_dir}")

  # Get channels
  channels = do.get_channels()
  info(f"  Channels: {channels}")

  # Get bounds for the first channel
  start, end = do.get_bounds(channels[0])
  info(f"  Channel start Unix time: {start}")
  info(f"  Channel stop Unix time:  {end}")

  properties = do.get_properties(channels[0])
  info("  Properties:")
  for key, value in properties.items():
    info(f"    {key}: {value}")

  sample_rate = properties['samples_per_second']
  msg = f"  Computed start UTC time: {datetime.utcfromtimestamp(start / int(sample_rate))}"
  info(msg)
  msg = f"  Computed stop UTC time:  {datetime.utcfromtimestamp(end / int(sample_rate))}"
  info(msg)

  # Get continuous blocks of data
  continuous_blocks = do.get_continuous_blocks(start, end, channels[0])
  plural = "s" if len(continuous_blocks) > 1 else ""
  info(f"\n  Found {len(continuous_blocks)} continuous block{plural}")

  metadata_samples = do.read_metadata(
        start_sample=start,
        end_sample=end,
        channel_name=channels[0],
    )

  all_data = []
  all_timestamps = []
  bn = 0

  # Read data blocks, but discard content after printing info about it.
  read_data = cli().read_data
  # Read and concatenate data blocks to return to caller.
  return_data = cli().return_data

  for block_start, block_length in continuous_blocks.items():
    bn = bn + 1

    # Get the start and end timestamps for the current block.
    # TODO: Need to use epoch as it may not always be Unix epoch as assumed below.
    start_time = datetime.utcfromtimestamp(block_start / int(sample_rate))
    end_time = start_time + timedelta(seconds=(int(block_length)-1) / int(sample_rate))
    info(f"    Block {bn}:")

    info(f"      Block start:    {block_start}")
    info(f"      Block length:   {block_length}")
    info(f"      Computed start: {start_time}")
    info(f"      Computed stop:  {end_time}")
    if block_start in metadata_samples:
      info("      Metadata:")
      for key, value in metadata_samples[block_start].items():
        info(f"        {key}: {value}")
    else:
      info("      Metadata: No metadata for this block.")


    if not read_data:
      info("      Data: Not read because read_data is set to False.")
    else:
      data = do.read_vector(block_start, block_length, channels[0])

      info("      Data:")
      info(f"        # samples:      {len(data)}")
      info(f"        data.shape:     {data.shape}")

      if return_data:
        all_data.append(data)
        all_timestamps.append((start_time, end_time, sample_rate))

  if return_data and len(all_data) > 0:
    all_data = np.concatenate(all_data)

  return all_timestamps, all_data


def process_observations(station_dir, n=None):
  """Process all subdirectories starting with 'OBS'"""

  info("")
  info(utilrsw.hline(display=False))
  info(f'Station directory: {station_dir}')

  obs_dirs = dirs(station_dir)

  found_obs_dir = False
  n_processed = 0
  for obs_dir in obs_dirs:
    if n is not None and n_processed >= n:
      break
    if obs_dir.startswith('OBS'):
      found_obs_dir = True
      dir_path = os.path.join(station_dir, obs_dir)

      try:
        import time
        time_read1 = time.time()
        time_, data = process_observation(dir_path)
        dt1 = time.time() - time_read1
        log.info(f"Time to read dir content: {dt1:.4f} seconds")

        station_id = os.path.basename(station_dir)
        cache_file = os.path.join("/tmp/cache", station_id, obs_dir) + ".pkl"
        utilrsw.write(cache_file, {"time": time_, "data": data})

        time_read2 = time.time()
        utilrsw.read(cache_file)
        log.info(f"Time to read cached dir content: {time.time() - time_read2:.4f} seconds")
        dt2 = time.time() - time_read2
        log.info(f"Speedup from caching: {dt1/dt2:.2f}x")

        n_processed += 1
      except Exception as e:
        error(f"Error processing {dir_path}: {e}")

  if not found_obs_dir:
    info(f"No 'OBS' directories found in {station_dir}")


if __name__ == '__main__':
  args = cli()

  # Process all station directories.
  station_dirs = dirs(args.base_dir)
  for station_dir in station_dirs:
    if args.station is not None and station_dir != args.station:
      continue
    station_dir = os.path.join(args.base_dir, station_dir)
    process_observations(station_dir, args.n)


if False:
  def extract_zip(zip_file):
    import zipfile

    dir_name = zip_file.replace('.zip', '')
    os.makedirs(dir_name, exist_ok=True)

    # Extract the contents of the zip file into the directory
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(dir_name)

    return dir_name

  import sys
  # Old code for processing zip files downloaded from PSWS web page.
  zip_dir = sys.argv[1]
  files = os.listdir(zip_dir)
  files.sort()
  for file in files:
    if file.endswith('.zip'):
      zip_file = os.path.join(zip_dir, file)
      dir_name = extract_zip(zip_file)
      time_, data = process_observation(dir_name)
