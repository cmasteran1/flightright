# src/flightright/config/weather.py

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# NWS (optional fallback)
NWS_BASE_URL = "https://api.weather.gov"

# Retry policy
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0