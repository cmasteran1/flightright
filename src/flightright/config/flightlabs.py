# src/flightright/config/flightlabs.py

from flightright.config.env import FLIGHTLABS_ACCESS_KEY

BASE_URL = "https://www.goflightlabs.com"

DEFAULT_LIMIT = 500     # large enough to capture peak airports
REQUEST_TIMEOUT = 20.0

ACCESS_KEY = FLIGHTLABS_ACCESS_KEY