from bilstm_with_attention import attention_bilstm

TIME_STEPS = 28
INPUT_DIM = 28
LSTM_UNITS = 64

model = attention_bilstm(time_steps=TIME_STEPS, input_dim=INPUT_DIM, lstm_units=LSTM_UNITS)
