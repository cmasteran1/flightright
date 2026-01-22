#!/bin/bash

export PYTHONPATH="$PWD/src:$PYTHONPATH"

python - <<'PY'
from runtime.providers.weather.nws import NWSProvider
# JFK approx
lat, lon = 40.6413, -73.7781
prov = NWSProvider()
print(prov.get_features(lat, lon, "2026-01-22", 20))  # local date + local dep hour
PY
