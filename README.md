![alt text](https://github.com/lucasglanville/and_theyre_off_backend/assets/123101163/af2fcd24-8dd3-401e-a802-71a508e3a723 "Logo")

# AND THEY'RE OFF 

[Website](https://andtheyreoff.streamlit.app/)


Welcome to AND THEY'RE OFF, a data-science project that uses a deep learning approach for predicting the age-old sport of horse racing.
Our team set out to create an innovative model capable of forecasting the profitability of each horse in any given Flat Handicap race.

### Using Neural Networks to analyse historic horse-racing data

Our model was trained on horse racing data from 2020-2022 and tested on data from 2022-2023.
The raw data included basic features such as previous wins, starting odds, jockey and trainer, to name a few. However, we soon realised that original features were not enough for our model to consistently profit against Betfair Exchange odds and commissions during the test stage.
It was our engineered new features and custom loss function, all wrapped in a Deep Learning Neural Network that gave us our perceived edge and what we set out to achieve. The output is now a sophisticated model that we can feed daily pre-race data to generate a model confidence metric. This metric, plus our strategy, results in a 'back' or 'threshold not met' decision that can be seen in the 'Predict' feature on the 'races' tab on our website.
Our model and strategy resulted in a 29% ROI over 1000 simulated backings in the one year test period.




### Simulated results

Here are our simulated profits of betting £1 on each horse that our model indicates. To compare, we also show the results of betting £1 every race on the horse with the best odds, betting £1 on every horse, and one of the best strategies there is, not betting at all:

![alt text](https://github-production-user-asset-6210df.s3.amazonaws.com/123101163/267362395-21dafab8-58d6-4acc-821a-a3705cce5acf.png "Returns vs baselines")

Unsurprisingly, betting on every horse is a sure-fire way to lose all your money. Betting on the horse with the best odds appears to be a damage-limiting strategy, making a small loss. Our neural network model is profitable over this test set, and over a reasonably large sample size too - it places 999 total bets over the 4150 races.
