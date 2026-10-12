source /Users/weigel/anaconda3/etc/profile.d/conda.sh; conda activate
conda activate python3.10.9-cdawmeta
tableui-serve --config ../table/tableui.json --host 0.0.0.0 --port 8053 --workers 4
