cd /home/bengelke/hapi/digital-rf
git pull
pip install -e . --force-reinstall

# Run analysis on all DRF observations dirs for all stations
cd /home/bengelke/hapi/digital-rf
python3 drf.py --station-dir /home --use-cache --cache-samples --n -1

# Copy results to Bob's server. If needed, password is F...S...S...8675
rsync -avz data guest1@rweigel.dynu.net:digital-rf
