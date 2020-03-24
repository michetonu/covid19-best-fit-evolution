from os.path import dirname, realpath

DATA_URL = "https://raw.githubusercontent.com/pomber/covid19/master/docs/timeseries.json"

# Number of days to plot
MAX_DAYS_AHEAD = 30

# Minimum of confirmed cases from which to start plotting
MIN_CONFIRMED_CASES = 20

MIN_DEATHS = 2

SRC_PATH = dirname(realpath(__file__))
