# Daily Updates

Update HAPI `digital-rf` code, execute metadata run, and push tables and log to HAPI `digital-rf` repository.

```
cd /home/bengelke/hapi/digital-rf
./drf.sh
```

View results (see Install section below if above was run on headless server).

```
tableui-serve --conf conf/tableui.json
```

# Install

If directory `/home/bengelke/hapi/digital-rf` does not exist,

```
cd /home/bengelke/hapi
git clone https://github.com/rweigel/digital-rf
cd /home/bengelke/hapi/digital-rf
pip install -e . --force-reinstall
```

View table from last run
```
tableui-serve --conf conf/tableui.json
```

# Update `digital-rf`

```
cd /home/bengelke/hapi/digital-rf
git pull
pip install -e . --force-reinstall
``

# View Existing Table from Previous Run

```
cd /home/bengelke/hapi/digital-rf
tableui-serve --conf conf/tableui.json
```

# Run

```
# Run analysis on all DRF observations dirs for all stations
cd /home/bengelke/hapi/digital-rf
python3 drf.py --station-dir /home --use-cache --cache-samples --n -1

# Copy results to Bob's server. If needed, password is F...S...S...8675
rsync -avz table guest1@rweigel.dynu.net:digital-rf
rsync -avz cache guest1@rweigel.dynu.net:digital-rf
rsync -avz log guest1@rweigel.dynu.net:digital-rf

# Optionally view table from analysis
tableui-serve --conf conf/tableui.json
```

# Debug

```
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
