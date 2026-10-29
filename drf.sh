#!/bin/bash

if [[ "$(id -un)" == "benglke" ]]; then
	cd /home/bengelke/hapi/digital-rf || exit 1
	git pull
	pip install -e . --force-reinstall
else
  cd /home/rweigel/digital-rf || exit 1
	source .venv-drf/bin/activate || exit 1
fi

# Run analysis on all DRF observations dirs for all stations
python3 drf.py --station-dir /home --use-cache --cache-samples --n -1

# Copy results to Bob's server.
rsync -avz data guest1@rweigel.dynu.net:digital-rf

git commit -m "Update tables" -- data/table
git commit -m "Update log" -- data/log
git push
