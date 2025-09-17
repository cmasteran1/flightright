FlightRight is a product which allows consumers to make more informed decisions about their flight based travel.
This is achieved by providing accurate up-to-date modeling about the delay probabilities of US domestic flights.
FlightRight has 2 main arms, model design and model deploy. In the model design phase, publicly available data is gathered,
parsed, and pruned. Then data is feed into a corresponding ML training pipeline. There exist multiple ML models for various
possible predictions that may be made. In the development phase, various models will be tested for efficacy in solving the same problem.
In the deployment phase, different models will be needed to answer different questions. For example; what is the estimated delay time 7 days from my flight?
what is the estimated delay time 3 hours from my flight? To answer the second question, we will have vast quantities of additional information.
The airline may already have added a delay, the airport may already be showing signs of congestion/stress, The en-route plan may already be delayed.

    In the deployed model, the following basic features must exist: estimated delay time (with proper bounds) whenever the customer makes a request, and
    secondarily, a percentage chance of unexpected major delay. This is because certain factors may be extremely non-linear and difficult to predict such as:
    major equipment failure, unexpected major weather systems, absent crew, and major airline operational failure. For example, it could be
    a clear but busy day and the customer's plane experiences engine problems on the runway. We would hope that the model would actually predict
    an on-time flight, however, provide a percentage chance that the flight is delayed by > 30 minutes.