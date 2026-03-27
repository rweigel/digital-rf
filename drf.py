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
      '--first-last', action='store_true', dest='first_last',
      help='Only process the first and last OBS directory for each station.'
  )
  parser.add_argument(
      '--start', type=str, default=None,
      help='UTC start time (inclusive) in format yyyy-mm-ddThh:mm:ss:...Z, ... is nanoseconds. If one of --start/--stop is given, both required.'
  )
  parser.add_argument(
      '--stop', type=str, default=None,
      help='UTC stop time (inclusive) in format yyyy-mm-ddThh:mm:ss:...Z, ... is nanoseconds. If one of --start/--stop is given, both required.'
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

  # If one of start/stop is given, require both
  if (args.start is not None) != (args.stop is not None):
    parser.error('If one of --start/--stop is given, both are required.')

  # If start/stop given and n not set by user, set n=-1 (process all)
  import sys
  n_set = '--n' in sys.argv
  if args.start is not None and not n_set:
    args.n = -1

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
  from datetime import datetime, timedelta, timezone

  import digital_rf as drf

  do = drf.DigitalRFReader(observation_dir)

  info(f"Reading: {observation_dir}")

  # Get channels
  channels = do.get_channels()
  info(f"  Channels: {channels}")

  # Get bounds for the first channel
  start_sample, end_sample = do.get_bounds(channels[0])
  info(f"  Channel start index: {start_sample}")
  info(f"  Channel stop index:  {end_sample}")

  properties = do.get_properties(channels[0])
  info("  Properties:")
  for key, value in properties.items():
    info(f"    {key}: {value}")

  epoch = properties.get('epoch', '1970-01-01T00:00:00Z')
  try:
    epoch_dt = datetime.strptime(epoch, '%Y-%m-%dT%H:%M:%S%z')
    epoch_unix = epoch_dt.timestamp()
  except Exception as e:
    error(f"Error parsing epoch: {epoch}. Error: {e}")
    return None

  start_sample_epoch = start_sample / int(properties['samples_per_second'])
  end_sample_epoch = end_sample / int(properties['samples_per_second'])
  start_sample_utc = datetime.fromtimestamp(start_sample_epoch + epoch_unix, tz=timezone.utc)
  start_sample_utc = datetime.strftime(start_sample_utc, '%Y-%m-%dT%H:%M:%S.%fZ')
  end_sample_utc = datetime.fromtimestamp(end_sample_epoch + epoch_unix, tz=timezone.utc)
  end_sample_utc = datetime.strftime(end_sample_utc, '%Y-%m-%dT%H:%M:%S.%fZ')

  samples_per_second = properties['samples_per_second']
  msg = f"  Computed start seconds since epoch: {start_sample_epoch} = (start index)/samples_per_second"
  info(msg)
  msg = f"  Computed stop seconds since epoch:  {end_sample_epoch} = (end index)/samples_per_second"
  info(msg)
  msg = f"  Computed start UTC time: {start_sample_utc}"
  info(msg)
  msg = f"  Computed stop UTC time:  {end_sample_utc}"
  info(msg)

  # Get continuous blocks of data
  continuous_blocks = do.get_continuous_blocks(start_sample, end_sample, channels[0])
  plural = "s" if len(continuous_blocks) > 1 else ""
  info(f"\n  Found {len(continuous_blocks)} continuous block{plural}")

  metadata_samples = do.read_metadata(
    start_sample=start_sample,
    end_sample=end_sample,
    channel_name=channels[0])

  all_metadata = {
    'channels': channels,
    'start_sample': start_sample,
    'start_sample_utc': start_sample_utc,
    'end_sample': end_sample,
    'end_sample_utc': end_sample_utc,
    'properties': properties,
    'continuous_blocks_note': "continuous_blocks is a dict with keys of start unix time and values of block length in samples. The actual unix time range of the block can be computed from the start unix time and block length using the sample_rate property.",
    'continuous_blocks': continuous_blocks,
    'block_unix_time_ranges': [],
    'metadata_samples': metadata_samples
  }

  all_data = []
  all_times = []
  bn = 0

  # Read data blocks, but discard content after printing info about it.
  read_data = cli().read_data
  # Read and concatenate data blocks to return to caller.
  return_data = cli().return_data

  for block_start, block_length in continuous_blocks.items():
    bn = bn + 1

    # Get the start and end timestamps for the current block.
    # TODO: Need to use epoch as it may not always be Unix epoch as assumed below.
    start_time = datetime.fromtimestamp(block_start / int(samples_per_second), tz=timezone.utc)
    end_time = start_time + timedelta(seconds=(int(block_length)-1) / int(samples_per_second))
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
        block_unix_time_start = block_start / int(samples_per_second)
        block_unix_time_end = block_unix_time_start + (block_length - 1) / int(samples_per_second)
        all_metadata['block_unix_time_ranges'].append((block_unix_time_start, block_unix_time_end))
        all_data.append(data)
        times = np.arange(block_start, block_start + block_length) / int(samples_per_second)
        all_times.append(times)

  if return_data and len(all_data) > 0:
    all_data = np.concatenate(all_data)

  return all_times, all_data, all_metadata

def _parse_obs_time(d, pat):
  import datetime
  m = pat.match(d)
  if not m:
    return None
  s = m.group(1).replace('-', ':', 2).replace('-', ':', 1).replace('T', 'T', 1)
  # s is now '2024-02-04T00:00'
  try:
    return datetime.strptime(s, '%Y-%m-%dT%H:%M')
  except Exception:
    return None


def _parse_cli_time(ts):
  import datetime
  # Accepts yyyy-mm-ddThh:mm:ss:...Z, ... is nanoseconds
  # We'll parse up to seconds, ignore nanoseconds and Z
  ts = ts.rstrip('Z')
  base = ts.split(':')[0:3]  # yyyy-mm-ddThh, mm, ss
  base = ':'.join(base)
  try:
    return datetime.strptime(base, '%Y-%m-%dT%H:%M:%S')
  except Exception:
    try:
      return datetime.strptime(base, '%Y-%m-%dT%H:%M')
    except Exception:
      return None


def _subset_obs_dirs(station_dir, n=None, first_last=False, start=None, stop=None):
  import re
  obs_dirs = [d for d in dirs(station_dir) if d.startswith('OBS')]
  pat = re.compile(r'^OBS(\d{4}-\d{2}-\d{2}T\d{2}-\d{2})')

  # If start/stop given, filter obs_dirs by timestamp in filename
  if start is not None and stop is not None:
    start_dt = _parse_cli_time(start)
    stop_dt = _parse_cli_time(stop)
    if start_dt is not None and stop_dt is not None:
      filtered = []
      for d in obs_dirs:
        dt = _parse_obs_time(d, pat)
        if dt is not None and start_dt <= dt <= stop_dt:
          filtered.append(d)
      info(f'Filtering OBS dirs by start/stop: {start} to {stop}. {len(filtered)} remain.')
      obs_dirs = filtered

  if first_last and len(obs_dirs) > 0:
    k = n if n is not None else 1
    first_n = obs_dirs[:k]
    last_n  = obs_dirs[-k:]
    # Combine preserving order and avoiding duplicates
    seen = set()
    selected = []
    for d in first_n + last_n:
      if d not in seen:
        seen.add(d)
        selected.append(d)
    info(f'--first-last (n={k}): selecting {selected}')
    obs_dirs = selected

  return obs_dirs


def process_observations(station_dir, n=None, first_last=False, start=None, stop=None):
  """Process all subdirectories starting with 'OBS'"""
  import time

  info("")
  info(utilrsw.hline(display=False))
  info(f'Station directory: {station_dir}')

  obs_dirs = _subset_obs_dirs(station_dir, n=n, first_last=first_last, start=start, stop=stop)

  found_obs_dir = False
  n_processed = 0
  obs_times = []
  obs_data = []
  obs_metadata = []
  for obs_dir in obs_dirs:
    if not first_last and n is not None and n_processed >= n:
      break
    found_obs_dir = True
    dir_path = os.path.join(station_dir, obs_dir)

    try:
      time_read1 = time.time()
      block_times, block_data, block_metadata = process_observation(dir_path)
      dt1 = time.time() - time_read1
      log.info(f"Time to read dir content: {dt1:.4f} seconds")

      obs_times.append(block_times)
      obs_data.append(block_data)
      obs_metadata.append(block_metadata)

      station_id = os.path.basename(station_dir)
      cache_file = os.path.join("/tmp/cache", station_id, obs_dir) + ".pkl"
      utilrsw.write(cache_file, {"time": block_times, "data": block_data, "metadata": block_metadata})

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
    return None
  else:
    return { "metadata": obs_metadata, "times": obs_times, "data": obs_data}

if __name__ == '__main__':
  args = cli()

  catalog = []
  # Process all station directories.
  station_dirs = dirs(args.base_dir)
  for station_dir in station_dirs:
    if args.station is not None and station_dir != args.station:
      continue
    catalog.append(station_dir)
    catalog.append("") # Nickname not available

    station_dir = os.path.join(args.base_dir, station_dir)
    result = process_observations(station_dir, n=args.n, first_last=args.first_last, start=args.start, stop=args.stop)

    if result is not None:
      startDateTime = result['metadata'][0]['start_sample_utc']
      catalog.append(startDateTime)
      stopDateTime = result['metadata'][-1]['end_sample_utc']
      catalog.append(stopDateTime)
      lat = result['metadata'][0]['metadata_samples'].get('lat', '')
      catalog.append(lat)
      long = result['metadata'][0]['metadata_samples'].get('long', '')
      catalog.append(long)
      elevation = "" # Not available in metadata_samples
      catalog.append(elevation)

      log.info(utilrsw.hline(display=False))
      log.info(f"Output for station directory: {station_dir}")
      log.info(utilrsw.hline(display=False))

      for idx, obs in enumerate(result['metadata']):
        log.info("Observation metadata:")
        log.info(utilrsw.format_dict(obs))
      if result['times'] is not None:
        log.info("Observation times:")
        log.info(result['times'])

  if len(catalog) > 0:
    utilrsw.write("metadata/drf/catalog.csv", catalog)

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
