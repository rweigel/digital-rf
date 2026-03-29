import os

import utilrsw

"""
Terminology:
  * Station: A station is a directory containing one or more samples of data.
  * Sample: A sample is a directory with a name of in the form
    OBSyyyy-mm-ddThh-mm, where yyyy-mm-ddThh-mm is the UTC time of the start of
    the sample. A sample contains one or more continuous blocks of data.
  * Block: A continuous segment of data. The segment is continuous in the sense
    that the measurements are at a regular cadence with no gaps.
  * HAPI Dataset: A set of blocks with the same samples_per_second,
    center_frequencies, grid_square, uuid_str, lat, and long.

    The HAPI dataset ID is the station directory name (e.g., S000042).

    Some stations have multiple HAPI datasets. For example, station S000042
    has blocks with 9 or 10 center frequencies. This code only forms one HAPI
    dataset per station. It does this by only using blocks for which the block
    metadata matches that for the first block in the first sample. Although it
    is possible to form multiple HAPI datasets per station by grouping blocks
    by common metadata, this is not done because it would require scanning all
    files.
"""

base_dir = utilrsw.script_info()['dir']
LOG_DIR = os.path.join(base_dir, "log", "drf")
META_DIR = os.path.join(base_dir, "metadata", "drf")
CACHE_DIR = os.path.join("tmp", "cache", "drf")

format = u"%(message)s"
log = utilrsw.logger("drf", log_dir=LOG_DIR, console_format=format, file_format=format)


def process_samples(station_id, station_dir,
                    n=None,
                    first_last=False,
                    start=None,
                    stop=None,
                    read_data=False,
                    return_data=False,
                    cache_data=False,
                    cache_dir=CACHE_DIR):

  """Process all subdirectories starting with 'OBS'"""
  import time
  import numpy as np

  station_dir = os.path.join(station_dir, station_id)
  logger.info(station_id, "")
  logger.info(station_id, utilrsw.hline(display=False))
  logger.info(station_id, station_id)
  logger.info(station_id, f'  Station directory: {station_dir}')

  sample_dirs = _subset_sample_dirs(station_dir, n=n, first_last=first_last, start=start, stop=stop)

  found_station_dir = False
  n_processed = 0
  sample_times = []
  sample_data = []
  sample_metadata = {}

  metadata_first = None
  for sample_dir in sample_dirs:
    logger.info(station_id, "")

    if not first_last and n is not None and n_processed >= n:
      break
    found_station_dir = True
    dir_path = os.path.join(station_dir, sample_dir)

    try:

      time_read1 = time.time()
      times, data, metadata = _process_sample(station_id, dir_path, read_data=read_data, return_data=return_data)
      dt1 = time.time() - time_read1
      logger.info(station_id, f"  Time to read sample: {dt1:.4f} seconds")

      if metadata_first is None:
        metadata_first = metadata
      else:
        p1 = metadata_first['properties']
        p2 = metadata['properties']
        logger.info(station_id, "  Comparing sample properties with first sample properties:")
        _compare_sample_properties(station_id, p1, p2)

        block_metadata_first_key, block_metadata_first = next(iter(metadata_first['sample_block_metadata'].items()))
        msg = f"  Comparing first block metadata of this sample ({sample_dir}) with that from first block in the first sample ({block_metadata_first_key})."
        logger.info(station_id, msg)
        # TODO: Only need to compare metadata for first block of each sample b/c
        # we have previously compared block metadata within each sample.
        same = _compare_block_metadata(station_id, block_metadata_first, metadata['sample_block_metadata'])
        if same:
          logger.info(station_id, "    No block metadata differences found.")
        else:
          msg = f"    Skipping sample {sample_dir} because of block metadata difference from first block in first sample."
          logger.warning(station_id, msg)
          continue

      sample_times.append(times)
      sample_data.append(data)
      sample_metadata[sample_dir] = metadata

      if cache_data:
        station_id = os.path.basename(station_dir)
        cache_file = os.path.join(cache_dir, station_id, sample_dir) + ".pkl"
        cache_data = {
          "time": sample_times,
          "data": sample_data,
          "metadata": sample_metadata
        }
        utilrsw.write(cache_file, cache_data)

        time_read2 = time.time()
        utilrsw.read(cache_file)
        msg = f"  Time to read cached sample: {time.time() - time_read2:.4f} seconds"
        logger.info(station_id, msg)
        dt2 = time.time() - time_read2
        logger.info(station_id, f"  Speedup from caching: {dt1/dt2:.2f}x")

      n_processed += 1
    except Exception as e:
      logger.error(station_id, f"  Error processing {dir_path}: {e}")

  if not found_station_dir:
    logger.info(station_id, f"  No 'OBS' directories found in {station_dir}")
    return None
  else:
    # Each element of list obs_times is a NumPy array of times for all blocks
    # in an OBS directory.
    # Flatten to a single array of times.
    sample_times = np.concatenate(sample_times) if sample_times else np.array([])
    sample_data = np.concatenate(sample_data) if sample_data else np.array([])
    return { "metadata": sample_metadata, "times": sample_times, "data": sample_data}


def _process_sample(station_id, observation_dir, read_data=False, return_data=False):
  import numpy as np
  from datetime import datetime, timedelta, timezone

  import digital_rf as drf

  do = drf.DigitalRFReader(observation_dir)

  logger.info(station_id, f"  Sample: {os.path.basename(observation_dir)}")

  # Get channels
  channels = do.get_channels()
  logger.info(station_id, f"    Channels: {channels}")

  # Get bounds for the first channel
  start_sample, end_sample = do.get_bounds(channels[0])
  logger.info(station_id, f"    Channel start index: {start_sample}")
  logger.info(station_id, f"    Channel stop index:  {end_sample}")

  properties = do.get_properties(channels[0])
  logger.info(station_id, "    Properties:")
  for key, value in properties.items():
    logger.info(station_id, f"      {key}: {value}")

  epoch = properties.get('epoch', '1970-01-01T00:00:00Z')
  try:
    epoch_dt = datetime.strptime(epoch, '%Y-%m-%dT%H:%M:%S%z')
    epoch_unix = epoch_dt.timestamp()
  except Exception as e:
    logger.error(station_id, f"Error parsing epoch: {epoch}. Error: {e}")
    return None

  start_sample_epoch = start_sample / int(properties['samples_per_second'])
  end_sample_epoch = end_sample / int(properties['samples_per_second'])
  start_sample_utc = datetime.fromtimestamp(start_sample_epoch + epoch_unix, tz=timezone.utc)
  start_sample_utc = datetime.strftime(start_sample_utc, '%Y-%m-%dT%H:%M:%S.%fZ')
  end_sample_utc = datetime.fromtimestamp(end_sample_epoch + epoch_unix, tz=timezone.utc)
  end_sample_utc = datetime.strftime(end_sample_utc, '%Y-%m-%dT%H:%M:%S.%fZ')

  samples_per_second = properties['samples_per_second']
  msg = f"    Computed sample start seconds since epoch: {start_sample_epoch} = (start index)/samples_per_second"
  logger.info(station_id, msg)
  msg = f"    Computed sample stop seconds since epoch:  {end_sample_epoch} = (end index)/samples_per_second"
  logger.info(station_id, msg)
  msg = f"    Computed sample start UTC time: {start_sample_utc}"
  logger.info(station_id, msg)
  msg = f"    Computed sample stop UTC time:  {end_sample_utc}"
  logger.info(station_id, msg)

  # Get continuous blocks of data
  continuous_blocks = do.get_continuous_blocks(start_sample, end_sample, channels[0])
  plural = "s" if len(continuous_blocks) > 1 else ""
  logger.info(station_id, f"\n    Found {len(continuous_blocks)} continuous block{plural}")

  sample_block_metadata = do.read_metadata(start_sample=start_sample,
                                           end_sample=end_sample,
                                           channel_name=channels[0])

  sample_block_metadata_first = next(iter(sample_block_metadata.values()))

  metadata = {
    'channels': channels,
    'sample_block_metadata': sample_block_metadata,
    'start_sample': start_sample,
    'start_sample_utc': start_sample_utc,
    'end_sample': end_sample,
    'end_sample_utc': end_sample_utc,
    'properties': properties,
    'continuous_blocks_note': "continuous_blocks is a dict with keys of start block index and values of block length.",
    'continuous_blocks': continuous_blocks,
    'block_unix_time_ranges': [],
  }

  obs_data = []
  obs_times = []
  bn = 0

  for block_start, block_length in continuous_blocks.items():
    bn = bn + 1

    # Get the start and end timestamps for the current block.
    # TODO: Need to use epoch as it may not always be Unix epoch as assumed below.
    start_time = datetime.fromtimestamp(block_start / int(samples_per_second), tz=timezone.utc)
    end_time = start_time + timedelta(seconds=(int(block_length)-1) / int(samples_per_second))
    logger.info(station_id, f"      Block {bn}:")

    logger.info(station_id, f"        Block start:    {block_start}")
    logger.info(station_id, f"        Block length:   {block_length}")
    logger.info(station_id, f"        Computed start: {start_time}")
    logger.info(station_id, f"        Computed stop:  {end_time}")
    if block_start in sample_block_metadata:
      logger.info(station_id, "        Metadata:")
      for key, value in sample_block_metadata[block_start].items():
        logger.info(station_id, f"          {key}: {value}")
      logger.info(station_id, "        Comparing metadata for this block to that of the first block.")
      same = _compare_block_metadata(station_id, sample_block_metadata_first, sample_block_metadata, indent=10)
      if same:
        logger.info(station_id, "        Keeping block: No metadata differences found.")
      else:
        msg = "        Skipping block: Metadata for this block differs from that in first block."
        logger.warning(station_id, msg)
    else:
      logger.info(station_id, "        Metadata: No metadata for this block.")


    if not read_data:
      logger.info(station_id, "        Data: Not read because read_data is set to False.")
    else:
      data = do.read_vector(block_start, block_length, channels[0])

      logger.info(station_id,  "        Data:")
      logger.info(station_id, f"          # samples:      {len(data)}")
      logger.info(station_id, f"          data.shape:     {data.shape}")
      logger.info(station_id, f"          first 2 datum:  {data[0:2]}")
      logger.info(station_id, f"          last 2 datum:   {data[-2:]}")
      if return_data:
        block_unix_time_start = block_start / int(samples_per_second)
        block_unix_time_end = block_unix_time_start + (block_length - 1) / int(samples_per_second)
        metadata['block_unix_time_ranges'].append((block_unix_time_start, block_unix_time_end))
        obs_data.append(data)
        times = np.arange(block_start, block_start + block_length) / int(samples_per_second)
        obs_times.append(times)

  if return_data and len(obs_data) > 0:
    # Concatenate all sample blocks
    obs_data = np.concatenate(obs_data)
    obs_times = np.concatenate(obs_times)

  return obs_times, obs_data, metadata


class _log:
  def __init__(self, log_dir="/tmp/pwsw/log"):
    self.station_info_log = {}
    self.station_error_log = {}
    self.station_warning_log = {}
    self.log_dir = log_dir

  def _fmt_msg(self, msg):
    import io
    # Get message as string in the case that msg is a non-string object such
    # as a list, dict, or NumPy array.
    buf = io.StringIO()
    print(msg, file=buf)
    return buf.getvalue().rstrip()

  def info(self, station_id, msg):
    msg = self._fmt_msg(msg)
    log.info(msg)
    if station_id not in self.station_info_log:
      self.station_info_log[station_id] = []
    self.station_info_log[station_id].append(msg)

  def warning(self, station_id, msg):
    msg = self._fmt_msg(msg)
    log.warning(msg)
    if station_id not in self.station_warning_log:
      self.station_warning_log[station_id] = []
    self.station_warning_log[station_id].append(msg)

  def error(self, station_id, msg):
    msg = self._fmt_msg(msg)
    log.error(msg)
    if station_id not in self.station_error_log:
      self.station_error_log[station_id] = []
    self.station_error_log[station_id].append(msg)

  def write(self, station_id):
    # Write error log for station to file. Don't write if no errors.
    log_file = os.path.join(self.log_dir, "stations", f"{station_id}.error.log")
    log_lines = self.station_info_log.get(station_id, None)
    if log_lines is None:
      logger.info(station_id, f"  Writing: {log_file}")
      log_txt = "\n".join(log_lines)
    else:
      logger.info(station_id, f"  No errors; not writing error log file {log_file}")
      if os.path.exists(log_file):
        logger.info(station_id, f"No errors; removing old log file {log_file}")
        os.remove(log_file)

    # Write info log for station to file.
    log_file = os.path.join(self.log_dir, "stations", f"{station_id}.log")
    logger.info(station_id, f"  Writing: {log_file}")
    log_txt = "\n".join(self.station_info_log.get(station_id, ""))
    if "No 'OBS' directories" not in log_txt:
      utilrsw.write(log_file, log_txt)

    # Write warning log for station to file. Don't write if no warnings.
    log_file = os.path.join(self.log_dir, "stations", f"{station_id}.warning.log")
    log_lines = self.station_warning_log.get(station_id, None)
    if log_lines is not None:
      logger.info(station_id, f"  Writing: {log_file}")
      log_txt = "\n".join(log_lines)
      utilrsw.write(log_file, log_txt)


def _cli():
  import argparse

  parser = argparse.ArgumentParser(
      description='Read Digital RF data from PSWS station directories.'
  )
  parser.add_argument(
      '--station-dir', required=True, dest='station_dir',
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
      '--read-data', action='store_true', dest='read_data',
      help='Read data blocks. If not set, only metadata is read and data blocks are not read.'
  )
  parser.add_argument(
      '--return-data', action='store_true', dest='return_data',
      help='Return data blocks. If not set, data blocks are read but not concatenated and returned to caller.'
  )
  parser.add_argument(
      '--cache-data', action='store_true', dest='cache_data',
      help='Cache data for each sample.'
  )
  parser.add_argument(
      '--cache_dir', type=str, default="/tmp/cache", dest='cache_dir',
      help='Directory to use for caching sample data. Default is /tmp/cache.'
  )

  args = parser.parse_args()

  # If one of start/stop is given, require both
  if (args.start is not None) != (args.stop is not None):
    parser.logger.error(station_id, 'If one of --start/--stop is given, both are required.')

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


def _listdir(base_dir):
  dir_list = sorted(os.listdir(base_dir))
  # Keep only directories
  return [d for d in dir_list if os.path.isdir(os.path.join(base_dir, d))]


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


def _compare_block_metadata(station_id, metadata_ref, metadata, indent=4):

  import numpy as np

  same = True
  indent = " " * indent
  # Loop over all block metadata in first sample. We don't need the keys
  # (block start indices) for the comparison only the values
  # (metadata for each block).
  for block_start, block_metadata in metadata.items():
    for key, value in block_metadata.items():
      if key not in metadata_ref:
        same = False
        logger.info(station_id, f"{indent}Block metadata key '{key}' not found in first block metadata.")
        continue
      if isinstance(value, np.ndarray):
        if not np.array_equal(value, metadata_ref[key]):
          same = False
          logger.info(station_id, f"{indent}Block metadata key '{key}' has different values from that in first block metadata:")
          logger.info(station_id, f"{indent}  First:   {metadata_ref[key]}")
          logger.info(station_id, f"{indent}  Current: {value}")
      elif value != metadata_ref[key]:
        same = False
        logger.info(station_id, f"{indent}Block metadata key '{key}' has different values from that in first block metadata:")
        logger.info(station_id, f"{indent}  First:   {metadata_ref[key]}")
        logger.info(station_id, f"{indent}  Current: {value}")

  return same


def _compare_sample_properties(station_id, properties1, properties2):
  for key in properties1:
    if key not in properties2:
      logger.info(station_id, f"    Metadata key {key} not found in second metadata.")
    else:
      if properties1[key] != properties2[key]:
        logger.info(station_id, f"    Metadata key {key} has different values:")
        logger.info(station_id, f"      First:   {properties1[key]}")
        logger.info(station_id, f"      Current: {properties2[key]}")


def _subset_sample_dirs(station_dir, n=None, first_last=False, start=None, stop=None):
  import re
  obs_dirs = [d for d in _listdir(station_dir) if d.startswith('OBS')]
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
      logger.info(station_id, f'  {start} to {stop} gave {len(filtered)} sample dirs out of {len(obs_dirs)}.')
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
    logger.info(station_id, f'  first_last and n options gave {len(selected)} sample dirs out of {len(obs_dirs)}.')
    obs_dirs = selected

  return obs_dirs


def _catalog_entry(station_id, result):
  samples = list(result['metadata'].keys())
  if len(samples) == 0:
    logger.info(station_id, f"  No samples with metadata found for station directory {station_dir}. Cannot create catalog entry.")
    return ""

  catalog = []
  catalog.append(station_id)
  catalog.append("") # Nickname not available
  startDateTime = result['metadata'][samples[0]]['start_sample_utc']
  catalog.append(startDateTime)
  stopDateTime = result['metadata'][samples[-1]]['end_sample_utc']
  catalog.append(stopDateTime)

  sample_block_metadata = result['metadata'][samples[0]]['sample_block_metadata']
  if len(sample_block_metadata) == 0:
    msg = f"  No block metadata found for first sample {samples[0]}. Cannot create catalog entry."
    logger.info(station_id, msg)
    return ""

  sample_block_metadata_first = next(iter(sample_block_metadata.values()))
  lat = sample_block_metadata_first.get('lat', '')
  catalog.append(lat)
  long = sample_block_metadata_first.get('long', '')
  catalog.append(long)
  elevation = "" # Not available in obs_metadata
  catalog.append(elevation)

  return catalog


if __name__ == '__main__':
  args = _cli()
  logger = _log(log_dir=LOG_DIR)

  catalog = []
  for station_id in _listdir(args.station_dir):

    if args.station is not None and station_id != args.station:
      continue

    kwargs = {
      "station_dir": args.station_dir,
      "start": args.start,
      "stop": args.stop,
      "n": args.n,
      "first_last": args.first_last,
      "read_data": args.read_data,
      "return_data": args.return_data,
      "cache_dir": args.cache_dir,
      "cache_data": args.cache_data
    }
    result = process_samples(station_id, **kwargs)

    if result is not None:
      catalog.append(_catalog_entry(station_id, result))

      station_dir = os.path.join(args.station_dir, station_id)
      logger.info(station_id, utilrsw.hline(indent=2, display=False))
      logger.info(station_id, f"  Summary: {station_dir}")
      logger.info(station_id, utilrsw.hline(indent=2, display=False))

      logger.info(station_id, "    Samples kept:")
      for sample, sample_value in result['metadata'].items():
        logger.info(station_id, f"       {sample}")

      if result['times'] is not None:
        logger.info(station_id, f"    Dataset times (shape: {result['times'].shape}):")
        logger.info(station_id, f"      {result['times']}")
        logger.info(station_id, f"    Dataset data (shape: {result['data'].shape}):")
        logger.info(station_id, f"      {result['data']}")

    logger.info(station_id, utilrsw.hline(indent=2, display=False))
    logger.write(station_id)

  if len(catalog) > 0:
    fname = os.path.join(META_DIR, "catalog.csv")
    utilrsw.write(fname, catalog)


if False:
  def extract_zip(zip_file):
    import zipfile

    dir_name = zip_file.replace('.zip', '')
    os.make_dirs(dir_name, exist_ok=True)

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
      time_, data = _process_sample(dir_name)
