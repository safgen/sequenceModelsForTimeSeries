from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def get_vanilla_lstm(n_units=50):
    model = Sequential()
    model.add(LSTM(n_units, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def get_stacked_lstm(n_units=None):
    if n_units is None:
        n_units = [50, 50, 20]
    model = Sequential()
    model.add(LSTM(n_units[0], activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(n_units[1], activation='relu', return_sequences=True))
    model.add(LSTM(n_units[2], activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def get_bidirectional_lstm(n_units=50):
    model = Sequential()
    model.add(Bidirectional(LSTM(n_units, activation='relu'), input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = get_stacked_lstm()
# fit model
model.fit(X, y, batch_size=2, epochs=200, verbose=1)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
