from numpy import array
from basicLSTMs import get_stacked_lstm
from getSeqs import in_seq1 as raw_seq
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

EncoderDecoder = True


def get_encdec_model(n_steps_in, n_features, n_steps_out, n_units=100):
    model = Sequential()
    model.add(LSTM(n_units, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(n_units, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    return model


# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
if EncoderDecoder:
    y = y.reshape((y.shape[0], y.shape[1], n_features))
    model = get_encdec_model(n_steps_in, n_features, n_steps_out)
else:
    model = get_stacked_lstm(n_steps_in, n_features, n_outputs=n_steps_out)
# fit model
model.fit(X, y, epochs=200, verbose=1)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
