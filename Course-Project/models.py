from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Permute, Reshape)

def cnn_rnn_model(rows, cols, channel, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, rows, cols, channel))
    # Add convolutional layer
    conv_2d_1 = Conv2D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name = 'bn_conv_1')(conv_2d_1)
    # Add convolutional layer
    conv_2d_2 = Conv2D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv2')(bn_cnn)
    # Add batch normalization
    bn_cnn_out = BatchNormalization(name = 'bn_conv_2')(conv_2d_2)
    # Permute layer
    permuted = Permute(3, 2, 1)(bn_cnn_out)
    # Reshape layer
    reshaped = Reshape(cols, -1)(permuted)
    # Add a recurrent layer
    # Add bidirectional recurrent layer
    bidir_rnn_1 = Bidirectional(LSTM(units, 
                                    return_sequences=True, 
                                    activation='relu'), 
                                merge_mode='concat')(reshaped)
    # Add batch normalization
    bn_rnn = BatchNormalization(name = 'bn_rnn_1')(bidir_rnn_1)
    # Add bidirectional recurrent layer
    bidir_rnn_2 = Bidirectional(LSTM(units, return_sequences=True, activation='relu'), merge_mode='concat')(bn_rnn)
    # Add batch normalization
    bn_rnn_out = BatchNormalization(name = 'bn_rnn_2')(bidir_rnn_2)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_out)
    # Add softmax activation layer
    y_pred = Activation('softmax', name = 'softmax')(time_dense)
    # Specify the model
    model = Model(inputs = input_data, outputs = y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride