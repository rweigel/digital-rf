rsync -avz ../data/ weigel@rweigel.dynu.net:git/hapi/server-metadata/data/table

conda activate python3.10.9-cdawmeta

pkill -f "python serve.py --port 8051"
tableui-serve --port 8051 --config ../table/tableui.json &

sleep 1 # Wait for server to start
# Open browser to test
open http://localhost:8051
