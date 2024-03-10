# Our data
- hourly power production from 8 sources (coal, natural gas, nuclear, oil, solar, hydro, wind, other) from api 
- training data is tft/data/power_consumption_by_fuel_type.csv. 2018-07-01 to 2023-12-31
- inference is any time after 2023-12-31 to present time
- predictions are made for each source
- power_production.db is sqlite3 db containing most recent data up to present. not used for training only evaluation


# Model

## Temporal Fusion Transformers
- novel model for "Interpretable Multi-horizon Time Series Forecasting"

- interpretable: variable importance is measurable


## multi-horizon forecasting
- Multi-horizon forecasting refers to the process of predicting multiple future points in a time series, spanning over different time horizons. In contrast to single-step forecasting, where the focus is on predicting the next immediate value, multi-horizon forecasting aims to predict a sequence of future values 

### recursive or iterative
when making predictions for multiple time steps ahead of, each prediction is recursively fed back into the model to make subsequent predictions. this compounds error.

### direct or sequence to sequence 
use a fixed length of input (look-back window ) to make predictions for multiple time steps into the future at the same time, not relying on time steps directly previous

## types of variables 
each entity (individual moment in time) is associated with a set of variables.
- static covariates
- inputs
    - observed inputs (past)
    - known inputs (past and future)
- output (target value)
### static covariates
 - variables that do not change with time['fueltype', 'encoder_length', 'value_center', 'value_scale'] 
- these are fed into the model, same across all input values
### inputs
#### observed inputs
- these are measured at each time step and unknown beforehand. (target value)
- used for training but not inference

#### known inputs
- these are predetermined and generally not dependent on the observed inputs (time dependent in our case, all features derived from time, day-of-year, hour-of-day etc.)


### output
- target value, these are predicted for each group (fueltype) for each time step of prediction length(168 hours) for each quantile ([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]) using a fixed look-back length (config['pred_len']) 
- quantiles can be interpreted as the probability of the true value being below the estimated value at that quantile


## architecture

![tft](data/tft_architecture.png)

### variable selection network
- instance wise variable selection that learns which variables are most significant for each time step
- Beyond providing insights into which variables are most significant for the prediction problem, variable selection also allows TFT to remove any unnecessary noisy inputs which could negatively impact performance

#### input embedding
- represent each input variable from all variable types (static,observed and known) as higher dimensional tensor.
[time x variables x 1] becomes [time x variables x 512 ]
entity embedding for categorical and linear transformations for continuous.

- do learning learning on these to determine variables that contribute most variance and softmax to return probabilities which can be interpreted as variable importances

#### VSN
- concat all variable embeddings
- make 1 copy of input tensor to feed through following steps. another copy called RESIDUAL
    - linear(512,hidden_size)
    - elu(hidden_size) - activation 
    - linear(hidden_size,hidden_size)
    - glu(hidden_size) - activation
    - output = softmax(hidden_size) -> importance weights 
- output * RESIDUAL


### LSTM
Cell State: The central component of LSTM, acting as the network's memory, allows information to flow across time steps with minimal changes, enabling the network to maintain long-term dependencies.

Gates: LSTMs utilize three types of gates to regulate the flow of information into and out of the cell state, each serving a specific purpose

- Forget Gate: Decides what information to discard from the cell state, using a sigmoid function to weigh the importance of previous and current inputs.

- Input Gate: Determines which new information should be added to the cell state, combining a sigmoid function to select values and a tanh function to create a vector of new candidate values that could be added.

- Output Gate: Decides what part of the cell state should be output, again using a sigmoid function to select values and a tanh function to adjust the cell state, making it suitable for output.

An LSTM unit processes data sequentially, updating its cell state and generating an output at each time step. The gates within the LSTM selectively allow information to be added or removed from the cell state, enabling the network to "remember" relevant information over long sequences and "forget" irrelevant data. 


![lstm](data/lstm.png)

in our model we have a hidden size of 32 (32 LSTM units chained together) and 2 lstm layers. all of the weights are shared within each layer but the weights of the parameters in each cell are not all the same. 

- this is a good resource for implementing from scratch in Pytorch https://d2l.ai/chapter_recurrent-modern/lstm.html 

### Attention

This is very complicated. Read the paper and watch these videos for understanding.

Basically, take attention inputs (LSTM outputs), split into many smaller parts and do transformations on each then combine them back together. sounds very simple, but it works. 

An attention mechanism in a neural network, particularly in the context of models like Transformers, allows the model to focus on different parts of the input sequence when producing a specific part of the output sequence. It works by creating a weighted sum of all inputs, where the weights determine the focus or "attention" the model gives to each input at a given time

Query, Key, and Value Vectors: Each input element (e.g., a word in a sentence or each step in time) is represented by vectors. The model generates three vectors for each input: a query vector (Q), a key vector (K), and a value vector (V). These vectors are created by multiplying the input embeddings(from the variable selection network) by learned weight matrices



For our purposes, think of it as a black box. LSTM outputs go into the attention section and emerge with a richer representation of variables that the model can learn. 

- Paper: https://arxiv.org/pdf/1706.03762.pdf
- Explanation of theory and background(1 hr): https://www.youtube.com/watch?v=bCz4OMemCcA&list=LL&index=4&ab_channel=UmarJamil
- Implementation of Transformer in pytorch (3 hr): https://www.youtube.com/watch?v=ISNdQcPhsts&list=LL&index=3&ab_channel=UmarJamil 

![MH_attention](data/multiheadattentiojn.png)

Transformer (ChatGPT) has extra parts,  like different encoding, normalization, feed forward parts etc, but the attention mechanism is the same.  

The weights of the attention at each time step can be interpreted as the importance of that feature at that particular time step. 

### Output
- add skip connection from before attention which is LSTM output
- chained skip connections encode robustness 
- tensor that is size [8 (power sources), 168 (pred_len), 7(quantiles)]

- take 3rd item of last dimension of each step for 0.5 quantile prediction 

- sum all component predictions for total prediction 

## implementation 

- feature_creation.py contains feature engineering functions for each time step, and each group
    - instead of using date features directly, represent them as cycles. -> 24 is right next to 1 etc. 
    - day, 24
    - week, 7
    - day of year, 365
    - week of year, 52
    - month, 12
    - quarter, 4
![sin_time](data/sintime.png)

    - also have rolling mean/std for each fuel type
    - rolling mean/std for each time stamp (config['time_windows'])
    - shifted values for each fuel type. (config['large_time_windows'])

- tft.py, metrics.py, utils.py contains the classes needed for construction
    - tft.py contains most of the code from pytorch forecasting, rewritten for understanding 
    - there are some explanation notes in here 

- config.py contains parameters for model and trainer construction
    - sets general references to things that are constant across tft, trainer, and prediction. -> have to load in the model with the same params as it was trained on
    - max_pred_len is how long the predictions are 168 hours (1 week) 
    - max_encoder_len is how long the look back size is 
    - trainer_params is from lightning.pytorch - handles training loop


- model.py constructs training data and tft model
    - this is called by train.py to load in the model with the required params
    - time series dataset is a pytorch forecasting object that helps define the variable types and makes it easy to convert into pytorch dataloader. 
    - api -> csv/ sqlite database -> pd.DataFrame -> Feature creation ->  pytorch_forecasting.TimeSeriesDataSet -> torch.utils.data.DataLoader

- train.py trains with pytorch lightning trainer
    - this trains the model according to the config in config.py, configs of older runs are kept in checkpoint dir
    - ckpt_path -> include to continue training from a previous run, comment out to start new run
    - if __name__ == "__main__": is a common idiom used to determine whether a Python script is being run as the main program or if it has been imported as a module into another script

- predict.py contains the code to make predictions
    - this should be split into multiple functions for clarity
    - loads in model with the corresponding config used for training, important 
    - creates a dataframe with the correct look back length.
    - appends config['pred_len'] number of rows and includes the known future inputs and static covariates.
    - for the rolling values and shifted values (treated as known future inputs not sure why honestly), if they are inside the scope of the prediction window, the value from 1 year ago that hour is used. probably a better way to do this.

- make_predictions.ipynb is an easier place to view the predictions
    - prediction output is shape [8,168,7] (8 fuel types, 168 hours of predictions, 7 quantiles)

- checkpoints contains past configs and corresponding training runs 

- dev is workspace for thoughts and experimentation

# what to do
- more visualizations? show all of the quantiles as shaded region/confidence interval and true value
- get variable importances
- 