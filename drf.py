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

import os
import tempfile

import utilrsw


def process_samples(station_id, station_dir,
                    n=None,
                    first_last=False,
                    start_dt=None,
                    stop_dt=None,
                    read_samples=False,
                    return_samples=False,
                    cache_samples=False,
                    cache_dir=None):

  """
  Process all 'OBS' subdirectories in station_dir (user home directory).

  Each subdirectory starting with 'OBS' is referred to as a sample in this code.
  For each sample, this function reads all block metadata and optionally all
  data blocks.

  If `n` is given, only process up to `n` samples per station.

  If `first_last` is True, only process the first and last sample directories per
  station.

  If `start` and `stop` are given, only process samples with start times within
  the start/stop range (inclusive). The start time of a sample is determined
  by parsing the sample directory name, which has a name that starts with
  OBSyyyy-mm-ddThh-mm.

  If `read_samples` is True, read sample data blocks; otherwise only read metadata.

  If `return_samples` is True, return concatenated sample data blocks; otherwise
  that data from each block is discarded after reading (use for faster checking
  for files with errors).

  If `cache_samples` is True, cache the sample data blocks to files in `cache_dir`.
  """
  import time
  import numpy as np

  def compare_properties(metadata_first, metadata):

    p1 = metadata_first['sample']['properties']
    p2 = metadata['sample']['properties']
    logger.info(station_id, "  Comparing this sample's properties with first sample properties")
    _compare_sample_properties(station_id, p1, p2)

    bm_first_block_of_first_sample = utilrsw.get_path(metadata_first, ['blocks', 0, 'metadata'], default={})
    bm_first_block_of_current_sample = utilrsw.get_path(metadata_first, ['blocks', 0, 'metadata'], default={})
    msg = "  Comparing first block metadata of this sample with that from first block in the first sample."
    logger.info(station_id, msg)
    same = _compare_block_metadata(
      station_id,
      bm_first_block_of_first_sample,
      bm_first_block_of_current_sample)

    if same:
      logger.info(station_id, "    No block metadata differences found.")
    else:
      sample_id = os.path.basename(sample_dir)
      msg = f"    Skipping sample {sample_id} because its first block's metadata differs from that in the first block in the first sample."
      logger.warning(station_id, msg)
      return True

    return False

  def to_json_safe(obj):
    if isinstance(obj, np.ndarray):
      return [to_json_safe(i) for i in obj.tolist()]
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.generic):
      return obj.item()
    if isinstance(obj, dict):
      return {(k.item() if isinstance(k, np.generic) else k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
      return [to_json_safe(i) for i in obj]
    return obj

  def write_sample_cache(cache_dir, sample_dir, metadata, data):
    station_id = os.path.basename(station_dir)
    obs_dir = os.path.basename(sample_dir)

    cache_file = os.path.join(cache_dir, station_id, f"{obs_dir}.meta.pkl")
    logger.info(station_id, f"  Caching sample metadata to {cache_file}")
    utilrsw.write(cache_file, metadata)

    # Also write JSON file for debugging and inspection purposes.
    cache_file = os.path.join(cache_dir, station_id, f"{obs_dir}.meta.json")
    metadata_json = to_json_safe(metadata)
    utilrsw.write(cache_file, metadata_json)

    if data is not None:
      cache_file = os.path.join(cache_dir, station_id, f"{obs_dir}.data.pkl")
      logger.info(station_id, f"  Caching sample data to {cache_file}")
      utilrsw.write(cache_file, data)

  def read_sample_cache(cache_dir, sample_dir):
    station_id = os.path.basename(station_dir)
    obs_dir = os.path.basename(sample_dir)

    cached = {"metadata": None, "times": None, "data": None}

    cache_file = os.path.join(cache_dir, station_id, f"{obs_dir}.meta.pkl")
    logger.info(station_id, f"  Reading cached sample metadata from {cache_file}")
    tmp = utilrsw.read(cache_file)
    cached['metadata'] = tmp

    cache_file = os.path.join(cache_dir, station_id, f"{obs_dir}.data.pkl")
    if os.path.exists(cache_file):
      logger.info(station_id, f"  Reading cached sample data from {cache_file}")
      tmp = utilrsw.read(cache_file)
      cached = cached.update(tmp)

    return cached

  if cache_dir is None:
    cache_dir = os.path.join(_tmpdir(), "cache")

  station_dir = os.path.join(station_dir, station_id)
  logger.info(station_id, "")
  logger.info(station_id, utilrsw.hline(display=False))
  logger.info(station_id, station_id)
  logger.info(station_id, f'  Station directory: {station_dir}')

  # Determine which sample directories to process based on n, first_last, start, and stop.
  kwargs = {
    "n": n,
    "first_last": first_last,
    "start_dt": start_dt,
    "stop_dt": stop_dt
  }
  sample_names = _subset_sample_dirs(station_id, station_dir, **kwargs)

  sample_times = []
  sample_data = []
  sample_metadata = {}

  metadata_first = None
  found_obs_dir = False
  for sample_name in sample_names:
    logger.info(station_id, "")

    found_obs_dir = True
    sample_dir = os.path.join(station_dir, sample_name)

    try:

      kwargs = {
        "read_samples": read_samples,
        "return_samples": return_samples
      }
      time_read1 = time.time()
      times, data, metadata = _process_sample(station_id, sample_dir, **kwargs)
      dt1 = time.time() - time_read1
      logger.info(station_id, f"  Time to read sample: {dt1:.4f} seconds")

      sample_metadata[sample_name] = metadata

      if metadata_first is None:
        skip_sample = False
        metadata_first = metadata
      else:
        skip_sample = compare_properties(metadata_first, metadata)

      sample_metadata[sample_name]['skipped'] = skip_sample

      if cache_samples:
        cache_data = None
        if return_samples:
          cache_data = {"times": times, "data": data}
        write_sample_cache(cache_dir, sample_dir, metadata, cache_data)

        if False:
          # Eventually will add cli option use_cache to control whether to read.
          # We keep this here to enable benchmarking the speed of reading from
          # cache vs regular read.
          time_read2 = time.time()
          cache = read_sample_cache(cache_dir, sample_dir)
          dt2 = time.time() - time_read2
          logger.info(station_id, f"  Time to read cached sample: {dt2:.4f} seconds")
          logger.info(station_id, f"  Speedup from caching: {dt1/dt2:.2f}x")

      if skip_sample:
        continue

      sample_times.append(times)
      sample_data.append(data)

    except Exception as e:
      logger.error(station_id, f"  Error processing {sample_dir}: {e}")

  if not found_obs_dir:
    logger.info(station_id, f"  No 'OBS*' directories found in {station_dir}")
    return None
  else:
    # Each element of list sample_times is a NumPy array of times each block
    # in an OBS ('sample') directory. Flatten to a single array of times.
    sample_times = np.concatenate(sample_times) if sample_times else np.array([])
    sample_data = np.concatenate(sample_data) if sample_data else np.array([])
    result = {
      "metadata": sample_metadata,
      "times": sample_times,
      "data": sample_data
    }
    return result


def _process_sample(station_id, observation_dir, read_samples=False, return_samples=False):
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
    raise ValueError(f"Error parsing epoch: {epoch}. Error: {e}")

  start_sample_epoch = start_sample / int(properties['samples_per_second'])
  end_sample_epoch = end_sample / int(properties['samples_per_second'])

  start_sample_unix = start_sample_epoch + epoch_unix
  end_sample_unix = end_sample_epoch + epoch_unix

  start_sample_utc = datetime.fromtimestamp(start_sample_unix, tz=timezone.utc)
  start_sample_utc = datetime.strftime(start_sample_utc, '%Y-%m-%dT%H:%M:%S.%fZ')

  end_sample_utc = datetime.fromtimestamp(end_sample_unix, tz=timezone.utc)
  end_sample_utc = datetime.strftime(end_sample_utc, '%Y-%m-%dT%H:%M:%S.%fZ')

  end_sample_utc_exclusive = datetime.fromtimestamp(end_sample_unix + 1 / int(properties['samples_per_second']), tz=timezone.utc)
  end_sample_utc_exclusive = datetime.strftime(end_sample_utc_exclusive, '%Y-%m-%dT%H:%M:%S.%fZ')

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

  kwargs = {
    "channel_name": channels[0],
    "start_sample": start_sample,
    "end_sample": end_sample
  }
  sample_block_metadata = do.read_metadata(**kwargs)

  sample_block_metadata_first = _first(sample_block_metadata)

  metadata = {
    'sample': {
      'id': f"{station_id}/{os.path.basename(observation_dir)}",
      'channels': channels,
      'start_index': start_sample,
      'end_index': end_sample,
      'start_unix': start_sample_unix,
      'end_unix': end_sample_unix,
      'start_utc': start_sample_utc,
      'end_utc': end_sample_utc,
      'end_utc_exclusive': end_sample_utc_exclusive,
      'properties': properties,
    },
    'blocks': [],
    'n_blocks_skipped': 0,
  }

  sample_data = []
  sample_times = []
  bn = 0

  for block_start, block_length in continuous_blocks.items():
    bn = bn + 1

    block_info = {
      'start_index': block_start,
      'end_index': block_start + block_length - 1,
      'length': block_length
    }

    # Get the start and end timestamps for the current block.
    start_time = datetime.fromtimestamp(block_start / int(samples_per_second) + epoch_unix, tz=timezone.utc)
    end_time = start_time + timedelta(seconds=(int(block_length)-1) / int(samples_per_second))
    logger.info(station_id, f"      Block {bn}:")

    logger.info(station_id, f"        Block start:    {block_start}")
    logger.info(station_id, f"        Block length:   {block_length}")
    logger.info(station_id, f"        Computed start: {start_time}")
    logger.info(station_id, f"        Computed stop:  {end_time}")

    block_info['start_utc'] = datetime.strftime(start_time, '%Y-%m-%dT%H:%M:%S.%fZ')
    block_info['end_utc'] = datetime.strftime(end_time, '%Y-%m-%dT%H:%M:%S.%fZ')

    block_unix_time_start = block_start / int(samples_per_second)
    block_unix_time_end = block_unix_time_start + (block_length - 1) / int(samples_per_second)

    block_info['start_unix'] = block_unix_time_start
    block_info['end_unix'] = block_unix_time_end

    if block_start in sample_block_metadata:
      block_info['metadata'] = sample_block_metadata[block_start]


    if block_start not in sample_block_metadata:
      logger.info(station_id, "        Metadata: No metadata for this block.")
    else:
      logger.info(station_id, "        Metadata:")
      for key, value in sample_block_metadata[block_start].items():
        logger.info(station_id, f"          {key}: {value}")
      if bn > 1:
        logger.info(station_id, "        Comparing metadata for this block to that of the first block.")
        same = _compare_block_metadata(station_id, sample_block_metadata_first, sample_block_metadata[block_start], indent=10)
        if same:
          logger.info(station_id, "        Keeping block: No metadata differences found.")
        else:
          msg = f"        Skipping block with start {block_start}: Metadata for this block differs from that in first block."
          logger.warning(station_id, msg)
          metadata['n_blocks_skipped'] += 1
          block_info['skipped'] = True
          continue

    metadata['blocks'].append(block_info)

    if not read_samples:
      logger.info(station_id, "        Data: Not read because read_samples is set to False.")
    else:
      data = do.read_vector(block_start, block_length, channels[0])

      logger.info(station_id,  "        Data:")
      logger.info(station_id, f"          # samples:      {len(data)}")
      logger.info(station_id, f"          data.shape:     {data.shape}")
      logger.info(station_id, f"          first 2 datum:  {data[0:2]}")
      logger.info(station_id, f"          last 2 datum:   {data[-2:]}")

      if return_samples:
        sample_data.append(data)
        times = np.arange(block_start, block_start + block_length) / int(samples_per_second)
        sample_times.append(times)

  if return_samples and len(sample_data) > 0:
    # Concatenate all sample blocks
    sample_data = np.concatenate(sample_data)
    sample_times = np.concatenate(sample_times)

  return sample_times, sample_data, metadata


class _log:
  def __init__(self, log_dir=None):
    if log_dir is None:
      log_dir = os.path.join(_tmpdir(), "log")
    self.logs = {'info': {}, 'warning': {}, 'error': {}}
    self.log_dir = log_dir
    fmt = u"%(message)s"
    self.log = utilrsw.logger("drf", log_dir=log_dir, console_format=fmt, file_format=fmt)

  def _fmt_msg(self, msg):
    import io
    # Get message as string in the case that msg is a non-string object such
    # as a list, dict, or NumPy array.
    buf = io.StringIO()
    print(msg, file=buf)
    return buf.getvalue().rstrip()

  def _record(self, station_id, msg, level):
    msg = self._fmt_msg(msg)
    getattr(self.log, level)(msg)
    self.logs[level].setdefault(station_id, []).append(msg)

  def info(self, station_id, msg):
    self._record(station_id, msg, 'info')

  def warning(self, station_id, msg):
    self._record(station_id, msg, 'warning')

  def error(self, station_id, msg):
    self._record(station_id, msg, 'error')

  def _write_log_file(self, station_id, level, suffix, skip_if=None):
    log_file = os.path.join(self.log_dir, "stations", f"{station_id}.{suffix}")
    if os.path.exists(log_file):
      os.remove(log_file)
    log_lines = self.logs[level].get(station_id, None)

    if log_lines is not None:
      if level == 'warning':
        # Trim leading whitespace.
        log_lines = [line.lstrip() if isinstance(line, str) else line for line in log_lines]
        self.logs[level][station_id] = log_lines
      log_txt = "\n".join(log_lines)
      if skip_if is None or skip_if not in log_txt:
        self.log.info(f"  Writing: {log_file}")
        utilrsw.write(log_file, log_txt)
    return log_lines

  def write(self, station_id):
    self._write_log_file(station_id, 'error',   "error.log")
    self._write_log_file(station_id, 'info',    "log", skip_if="No 'OBS*' directories")
    self._write_log_file(station_id, 'warning', "warning.log")


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
      help='Only process the first and last n OBS directories for each station.'
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
      '--read-samples', action='store_true', dest='read_samples',
      help='Read sample data blocks. If not set, only metadata is read.'
  )
  parser.add_argument(
      '--return-samples', action='store_true', dest='return_samples',
      help='Return sample data blocks. If not set, sample data blocks are read and discarded.'
  )
  parser.add_argument(
      '--cache-samples', action='store_true', dest='cache_samples',
      help='Cache sample metadata (and data if --return-samples; a sample is a dir starting with OBS).'
  )

  base_dir = utilrsw.script_info()['dir']
  cache_dir = os.path.join(base_dir, "cache")
  log_dir = os.path.join(base_dir, "log", "drf")
  catalog_dir = os.path.join(base_dir, "metadata", "drf")

  parser.add_argument(
      '--cache-dir', type=str, default=cache_dir, dest='cache_dir',
      help=f'Directory to use for caching sample data. Default is {cache_dir}.'
  )
  parser.add_argument(
      '--log-dir', type=str, default=log_dir, dest='log_dir',
      help=f'Directory to use for caching sample data. Default is {log_dir}.'
  )
  parser.add_argument(
      '--catalog-dir', type=str, default=catalog_dir, dest='catalog_dir',
      help=f'Directory to use for caching sample data. Default is {catalog_dir}.'
  )

  args = parser.parse_args()

  # If one of start/stop is given, require both
  if (args.start is not None) != (args.stop is not None):
    exit('If one of --start/--stop is given, both are required.')

  # If start/stop given and n not set by user, set n=-1 (process all)
  import sys
  n_set = '--n' in sys.argv
  if args.start is not None and not n_set:
    args.n = -1

  if args.n == -1:
    args.n = None

  if args.return_samples:
    args.read_samples = True

  return args


def _first(dict_, key=None, default=None):
  """Return the first value in an ordered dict, or default if dict is empty.

  Examples:
    d = {'a': 1, 'b': {'x': 10, 'y': 20}, 'c': {'z': 100}}
    _first(d) -> 1
    _first(d, key='b') -> 10

    d = {'a': {'z': 100}}
    _first(d) -> {'z': 100}

  As of Python 3.7, regular dicts maintain insertion order, so this function
  will return the value of the first key inserted into the dict.
  """

  if not dict_:
    return {}
  if key is not None:
    if key not in dict_:
      return default
    dict_ = dict_[key]

  return next(iter(dict_.values()))


def _tmpdir():
  if os.path.exists("/tmp"):
    return "/tmp"
  else:
    return tempfile.gettempdir()


def _listdir(base_dir):
  dir_list = sorted(os.listdir(base_dir))
  # Keep only directories
  return [d for d in dir_list if os.path.isdir(os.path.join(base_dir, d))]


def _parse_obs_time(d, pat):
  import datetime
  m = pat.match(d)
  if not m:
    return None
  return datetime.datetime.strptime(m.group(1), '%Y-%m-%dT%H-%M')


def _parse_cli_time(ts):
  if ts is None:
    return None
  import datetime
  fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
  try:
    return datetime.datetime.strptime(ts, fmt)
  except Exception:
    exit(f'Error parsing command line time string {ts}. Expected format: {fmt}')


def _compare_block_metadata(station_id, metadata_ref, metadata, indent=4):

  import numpy as np

  same = True
  indent = " " * indent
  # Loop over all block metadata in first sample. We don't need the keys
  # (block start indices) for the comparison only the values
  # (metadata for each block).
  for key, value in metadata.items():
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


def _subset_sample_dirs(station_id, station_dir, n=None, first_last=False, start_dt=None, stop_dt=None):
  import re
  obs_dirs = [d for d in _listdir(station_dir) if d.startswith('OBS')]
  pat = re.compile(r'^OBS(\d{4}-\d{2}-\d{2}T\d{2}-\d{2})')

  # If start/stop given, filter obs_dirs by timestamp in filename
  if start_dt is not None and stop_dt is not None:
    filtered = []
    for obs_dir in obs_dirs:
      try:
        dt = _parse_obs_time(obs_dir, pat)
      except Exception:
        logger.error(station_id, f"Error parsing datetime from sample directory name {obs_dir}. Skipping dir. Expected format: {pat.pattern}")
        continue

      if dt is None:
        logger.error(station_id, f"  Could not parse datetime from sample directory name {obs_dir}.")
      if dt is not None and start_dt <= dt <= stop_dt:
        filtered.append(obs_dir)
    logger.info(station_id, f'  {start_dt} to {stop_dt} gave {len(filtered)} sample dirs out of {len(obs_dirs)}.')
    obs_dirs = filtered

  if len(obs_dirs) == 0:
    return []

  if not first_last:
    if n is not None:
      obs_dirs = obs_dirs[:n]
      logger.info(station_id, f'  first_last = False and n={n} gave {len(obs_dirs)} sample dirs out of {len(_listdir(station_dir))}.')

  if first_last:
    k = n if n is not None else 1
    first_n = obs_dirs[:k]
    last_n  = obs_dirs[-k:]
    # Combine preserving order and avoid duplicates
    seen = set()
    selected = []
    for d in first_n + last_n:
      if d not in seen:
        seen.add(d)
        selected.append(d)
    logger.info(station_id, f'  first_last = True and n={n} gave {len(obs_dirs)} sample dirs out of {len(_listdir(station_dir))}.')
    obs_dirs = selected

  return obs_dirs


def _catalog_entry(station_id, result):
  if result is None:
    return None
  samples = list(result['metadata'].keys())
  if len(samples) == 0:
    logger.info(station_id, f"  No samples with metadata found for station {station_id}. Cannot create catalog entry.")
    return None

  # Find index of last sample that is not skipped.
  last_idx = len(samples) - 1
  for idx, sample in enumerate(samples):
    if result['metadata'][sample].get('skipped', True):
      last_idx = idx - 1
      break
  if last_idx < 0:
    logger.info(station_id, f"  All samples were skipped for station {station_id}. Cannot create catalog entry.")
    return None

  catalog = []
  catalog.append(station_id)
  catalog.append("") # Nickname not available
  startDateTime = result['metadata'][samples[0]]['sample']['start_utc']
  catalog.append(startDateTime)
  stopDateTime = result['metadata'][samples[last_idx]]['sample']['end_utc_exclusive']
  catalog.append(stopDateTime)

  sample_block_metadata_first = result['metadata'][samples[0]]['blocks'][0].get('metadata', {})
  if len(sample_block_metadata_first) == 0:
    msg = f"  No block metadata found for first sample {samples[0]}. Cannot create catalog entry."
    logger.info(station_id, msg)
    return None

  lat = sample_block_metadata_first.get('lat', '')
  catalog.append(lat)
  long = sample_block_metadata_first.get('long', '')
  catalog.append(long)
  elevation = "" # Not available in obs_metadata
  catalog.append(elevation)

  return catalog


def _print_station_summary(station_id, result, station_dir, return_samples):
  import numpy as np

  if result is None:
    return

  station_dir = os.path.join(station_dir, station_id)
  logger.info(station_id, utilrsw.hline(indent=2, display=False))
  logger.info(station_id, f"  Summary: {station_dir}")
  logger.info(station_id, utilrsw.hline(indent=2, display=False))

  logger.info(station_id, "    Samples considered:")
  for sample_name, sample_value in result['metadata'].items():
    skipped = " (skipped)" if sample_value.get('skipped', False) else ""
    logger.info(station_id, f"      {sample_name}{skipped}")
    n_blocks = len(sample_value.get('blocks', -1))
    n_skipped = sample_value.get('n_blocks_skipped', 0)
    logger.info(station_id, f"        skipped blocks:     {n_skipped} of {n_blocks}")

    p = 'sample.properties.num_subchannels'
    num_subchannels = utilrsw.get_path(sample_value, p, 'None in metadata')
    logger.info(station_id, f"        num_subchannels:    {num_subchannels}")

    p = ['blocks', 0, 'metadata', 'center_frequencies']
    center_frequencies = utilrsw.get_path(sample_value, p, default=None)
    if isinstance(center_frequencies, np.ndarray):
      center_frequencies = np.array2string(center_frequencies, separator=', ', max_line_width=np.inf, formatter={'float_kind':lambda x:repr(float(x))})
    logger.info(station_id, f"        center_frequencies: {center_frequencies}")

  if return_samples:
    logger.info(station_id, f"    Dataset times (shape: {result['times'].shape}):")
    logger.info(station_id, f"      {result['times']}")
    logger.info(station_id, f"    Dataset data (shape: {result['data'].shape}):")
    logger.info(station_id, f"      {result['data']}")
  else:
    msg = "    Dataset times and data not returned because return_samples = False."
    logger.info(station_id, msg)


if __name__ == '__main__':

  args = _cli()
  logger = _log(log_dir=args.log_dir)

  print(f"Cache directory: {args.cache_dir}")
  print(f"Log directory:   {args.log_dir}")

  catalog = []
  for station_id in _listdir(args.station_dir):

    if args.station is not None and station_id != args.station:
      continue

    kwargs = {
      "station_dir": args.station_dir,
      "start_dt": _parse_cli_time(args.start),
      "stop_dt": _parse_cli_time(args.stop),
      "n": args.n,
      "first_last": args.first_last,
      "read_samples": args.read_samples,
      "return_samples": args.return_samples,
      "cache_samples": args.cache_samples,
      "cache_dir": args.cache_dir
    }

    result = process_samples(station_id, **kwargs)

    _print_station_summary(station_id, result, args.station_dir, args.return_samples)

    entry = _catalog_entry(station_id, result)
    if entry is not None:
      catalog.append(entry)

    logger.info(station_id, utilrsw.hline(indent=2, display=False))

    # Write info, warning, and error files.
    logger.write(station_id)

  if len(catalog) > 0:
    # Write HAPI catalog file
    fname = os.path.join(args.catalog_dir, "catalog.csv")
    print(f"\nWriting catalog to {fname}:")
    print("  station_id,nickname,startDateTime,stopDateTime,lat,long,elevation")
    for row in catalog:
      print("  " + ",".join(str(x) for x in row))
    utilrsw.write(fname, catalog)
