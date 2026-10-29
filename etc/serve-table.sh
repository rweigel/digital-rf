#!/bin/bash

source /Users/weigel/anaconda3/etc/profile.d/conda.sh; conda activate
conda activate python3.10.9-cdawmeta

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
(
	while true; do
		sleep 300
		git -C "$repo_dir" fetch origin &&
			git -C "$repo_dir" reset --hard '@{upstream}'
	done
) &
pull_pid=$!
trap 'kill "$pull_pid" 2>/dev/null; wait "$pull_pid" 2>/dev/null' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

tableui-serve --config ../table/tableui.json --host 0.0.0.0 --port 8053 --workers 4
