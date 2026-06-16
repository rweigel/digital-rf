cd /home/bengelke/hapi/digital-rf
git pull
pip install -e . --force-reinstall

# Run analysis on all DRF observations dirs for all stations
cd /home/bengelke/hapi/digital-rf
python3 drf.py --station-dir /home --use-cache --cache-samples --n -1

# Copy results to Bob's server.
rsync -avz data guest1@rweigel.dynu.net:digital-rf

git commit -m "Update tables" -- data/table
git commit -m "Update log" -- data/log