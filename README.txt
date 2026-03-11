    FlightRight is a product which allows consumers to make more informed decisions about their flight based travel.
This is achieved by providing accurate up-to-date modeling about the departure delay probabilities of US domestic flights.
FlightRight has 2 main arms, model design and model deploy. In the model design phase, publicly available data from BTS is gathered,
parsed, and pruned. Then data is feed into a corresponding ML training pipeline. There may exist multiple ML models for various
possible predictions that may be made. In the development phase, various models will be tested for efficacy in solving the same problem.
In the deployment phase, different models will be needed to answer the same question with varying amounts of data. For example; what chance does my flight have of being delayed by 30 minutes or more?
The amount of information available in the 7 days before the flight is different from the information 1 day before. 1 day before, we will have more accurate history, and likely more/accurate weather forcast data.
As of right now, we make no attempt to access pertinent data the day of the flight, thus these models should not be relied upon the day of travel.

    If the user wishes to build their own models using this pipline, they will start by running: python fetch_prune/fetch_all_flight.py --start YYYY/MM --end YYYY/MM
This grabs the historical BTS flight data for the months selected by the user. The BTS registry is a public database which provides ample information about
the history of commercial flights from registered US companies across all the U.S. Note that this will require a large amount of storage if more than a few months are
grabbed. Next, the user must modify a config file to decide how to prune the data. Since it is impractical to train on all the data
especially for multiple months, the user should filter by airline, airport, etc... other filters are present in the sample config dep_arr_config.json. This pruning step also adds weather data
obtained first from open-meteo free weather API and then cached locally for future use. The pruning step is run by: python src/fetch_prune/prepare_dataset.py path/to/config.json

    Now that we have pruned flights with attached weather, we can build specific features which might be viable for predicting our endpoint. As of now, we support building departure
delay features with plans to soon build arrival delay endpoint features as well. The training step is done using sklearn Isotonic Catboost Classifier networks. The current strategy is to
construct multiple binary classifiers for delays either exceeding a time threshold or failing to exceed that threshold. Separate classification data sets are then used to
construct multiple bin probabilities which correspond to the delay risk of a particular flight.


    In the deployed model, the following basic feature must exist: A % chance of delay in each time bin along with an explanation of the prediction.
The days out prediction model needs to scrape weather forcast results on demand, obtain recent historical trends for a given route,
and obtain expected congestion based on scheduled flights to and from origin and destination. These should be accessible with a minimal api resource usage.
The basic output should also support showing certain raw features in digestible human-readable form. Forecast data, route history, and scheduled congestion
are absolutely essential.

Currently, the deployment lives in flightright/src/flightright. Here, various features are pulled/computed mainly from flightlabs api endpoints. Weather forcast data can still
be pulled from open-meteo. The main workhorse is flightright/cli/predict.py. This executable verifies that a flight exists, based on the future flights endpoint. Then tries to find the recent flight history based
on the flight #. Then searches for daily and hourly forcast data. In the future, it will also search for airport and carrier history as soon as we can build up enough historical data to be meaningful.
Based on the available data, a model is chosen. The chosen model is then optionally found through external storage on a VM and downloaded. This can now output prediction bins as well as various meaningful features.

Users can now test out the deployment from a website: https://flightrightus.com/  . This site currently supports computing delay probability for a large amount of Southwest, American, Delta, and United airlines flights from popular US
destinations. Also reports various features that are used.
