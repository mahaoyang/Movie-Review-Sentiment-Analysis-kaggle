from keras import layers
from keras import optimizers
from keras.models import Model


def attention_3d_block(inputs, time_steps):
    # input_dim = int(inputs.shape[2])
    a = layers.Permute((2, 1))(inputs)
    a = layers.Dense(time_steps, activation='softmax')(a)
    a_probs = layers.Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = layers.multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def attention_bilstm(input_dim, time_steps, lstm_units):
    inputs = layers.Input(shape=(time_steps, input_dim))
    drop1 = layers.Dropout(0.3)(inputs)
    lstm_out = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
    attention_mul = attention_3d_block(lstm_out, time_steps)
    attention_flatten = layers.Flatten()(attention_mul)
    drop2 = layers.Dropout(0.3)(attention_flatten)
    output = layers.Dense(10, activation='sigmoid')(drop2)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    attention_bilstm(28, 28, 64)
