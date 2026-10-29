# Update all Metadata

Install and update HAPI `digital-rf` code, execute metadata run, and push tables and log to HAPI `digital-rf` repository:

```bash
cd /home/bengelke/hapi/digital-rf
./drf.sh
```

After this command is executed, log files, updated metadata, and tables file are visible at https://github.com/rweigel/digital-rf/tree/main/data/.

The `block.csv` and `samples.csv` tables are visible at https://hapi-server.org/meta/psws

# Development

## Install

See `drf.sh`

## Debugging

```bash
# View help
cd /home/bengelke/hapi/digital-rf
python3 drf.py --help

# Run analysis on first two DRF observation dirs in /home/S000182
cd /home/bengelke/hapi/digital-rf
python3 drf.py --station-dir /home --station S000182 --cache-samples --n 2

# Run analysis on first two DRF observation dirs for all stations
cd /home/bengelke/hapi/digital-rf
python3 drf.py --station-dir /home --cache-samples --n 2

# Run analysis on all DRF observations dirs for all stations
cd /home/bengelke/hapi/digital-rf
python3 drf.py --station-dir /home --cache-samples --n -1
```

## Remote Development

On pswsnetwork.eng.ua.edu,

```python
git clone https://github.com/rweigel/digital-rf
python3 -m venv .venv-drf
source .venv-drf/bin/activate
cd digital-rf/
pip install -e .
python drf.py --help
```

On local machine, one can execute `etc/mount.sh` to edit remote files using a local editor.

# Remaining HAPI Work

1. Determine what to put in HAPI catalog. Catalog will be derived from data/table/samples.csv (searchable at https://hapi-server.org/meta/digital-rf/samples)
2. Test workflow for updating HAPI catalog based on nightly uploads

# View Tables Locally

The `block` and `samples` tables visible at https://hapi-server.org/meta/psws can be viewed locally using

```bash
# From digital-rf directory
git pull 
pip install -e .
tableui-serve --conf conf/tableui.json
```
