    FlightRight is a product which allows consumers to make more informed decisions about their flight based travel.
This is achieved by providing accurate up-to-date modeling about the delay probabilities of US domestic flights.
FlightRight has 2 main arms, model design and model deploy. In the model design phase, publicly available data is gathered,
parsed, and pruned. Then data is feed into a corresponding ML training pipeline. There exist multiple ML models for various
possible predictions that may be made. In the development phase, various models will be tested for efficacy in solving the same problem.
In the deployment phase, different models will be needed to answer the same question at different times. For example; what chance does my flight have of being delayed by 30 minutes or more?
The amount of information available in the 7 days before the flight is different from the information available 3 hours before. 3 hours before, we will have vast quantities of additional information.
The airline may already have added a delay, the airport may already be showing signs of congestion/stress, the weather may have turned for the worse (or better). Thus, multiple models need to be built and maintained for
different situations.

    If the user wishes to build their own models using this pipline, they will start by running: python fetch_prune/fetch_all_flight.py --start YYYY/MM --end YYYY/MM
This grabs the historical BTS flight data for the months selected by the user. The BTS registry is a public database which provides ample information about
the history of commercial flights from registered US companies across all the U.S. Note that this will require a large amount of storage if more than a few months are
grabbed. Next, the user must modify the data/config.json file to decide which points they consider testing. Since it is impractical to train on all the data
especially for multiple months, the user should filter by airline, airport, etc... other filters present in config.json. This pruning step also adds weather data
obtained first from open-meteo free weather API and then cached locally for future use. The pruning step is run by: python build_training_set.py ../../config.json

    Now that we have pruned flights with attached weather, we can use the data to build a feature list and run a machine learning algorithm on the data
right now the days out model uses a catboost classifier algorithm however, developers will add additional models as they become relevant. At this point depending on the
model we want to construct, we can either build inbound prior data into the model or leave it out. The model will then undergo a calibration step whereby the bins should become
more accurate by fitting on an external set of data (handled by src/inference/predict_delay_bins.py).

    In the deployed model, the following basic feature must exist: A % chance of delay in each time bin along with an explanation of the prediction.
The days out prediction model needs to scrape weather forcast results on demand, obtain recent historical trends for a given route,
and obtain expected congestion based on scheduled flights to and from origin and destination.

the deployed model is still in development