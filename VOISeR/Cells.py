import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
import collections

#This is same structure to the LSTMStateTuple
_FeedbackStateTuple = collections.namedtuple("FeedbackStateTuple", ("h", "o"))
class FeedbackStateTuple(_FeedbackStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (h, o) = self
        if h.dtype != o.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" % (str(h.dtype), str(o.dtype)))
        return h.dtype

class FeedbackCell(RNNCell):
    def __init__(
        self,
        num_hidden_units,
        num_output_units,
        initializer= None,        
        use_bias = True,
        bias_initializer = tf.zeros_initializer,
        hidden_activation=None, #Baisc is tanh.        
        output_state_activation=None, #This is only used for state.
        state_is_tuple=True,
        reuse=None,
        name=None
        ):
        super(FeedbackCell, self).__init__(_reuse=reuse, name=name)
        self._num_hidden_units = num_hidden_units        
        self._num_output_units = num_output_units        
        self._initializer= initializer        
        self._use_bias = use_bias
        self._bias_initializer = bias_initializer
        self._hidden_activation = hidden_activation or tf.nn.tanh
        self._output_state_activation = output_state_activation or tf.nn.softmax
        self._reuse = reuse
        self._name = name

        if state_is_tuple:
            self._state_size = FeedbackStateTuple(num_hidden_units, num_output_units)
        else:
            self._state_size = num_hidden_units + num_output_units
        self._output_size = num_hidden_units + num_output_units

    @property
    def state_size(self):
        return self._state_size
    
    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        input_Size = inputs.get_shape().with_rank(2)[1]

        with tf.variable_scope(self._name or type(self).__name__):
            if input_Size.value is None:
                raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

            hidden = tf.layers.dense(
                inputs= tf.concat([inputs, state.h, state.o], axis = 1),
                units= self._num_hidden_units,
                activation= self._hidden_activation,
                use_bias= False,
                name= "hidden"
                )
            output = tf.layers.dense(
                inputs= hidden,
                units= self._num_output_units,
                use_bias= self._use_bias,
                name= "output"
                )

            new_State = FeedbackStateTuple(h=hidden, o=self._output_state_activation(output))

        return tf.concat([hidden, output], axis=1), new_State

class FeedbackOnlyCell(RNNCell):
    def __init__(
        self,
        num_hidden_units,
        num_output_units,
        initializer= None,        
        use_bias = True,
        bias_initializer = tf.zeros_initializer,
        hidden_activation=None, #Baisc is tanh.        
        output_state_activation=None, #This is only used for state.
        reuse=None,
        name=None
        ):
        super(FeedbackOnlyCell, self).__init__(_reuse=reuse, name=name)
        self._num_hidden_units = num_hidden_units        
        self._num_output_units = num_output_units        
        self._initializer= initializer        
        self._use_bias = use_bias
        self._bias_initializer = bias_initializer
        self._hidden_activation = hidden_activation or tf.nn.tanh
        self._output_state_activation = output_state_activation or tf.nn.softmax
        self._reuse = reuse
        self._name = name
        
        self._state_size = num_output_units
        self._output_size = num_hidden_units + num_output_units

    @property
    def state_size(self):
        return self._state_size
    
    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        input_Size = inputs.get_shape().with_rank(2)[1]

        with tf.variable_scope(self._name or type(self).__name__):
            if input_Size.value is None:
                raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

            hidden = tf.layers.dense(
                inputs= tf.concat([inputs, state], axis = 1),
                units= self._num_hidden_units,
                activation= self._hidden_activation,
                use_bias= False,
                name= "hidden"
                )
            output = tf.layers.dense(
                inputs= hidden,
                units= self._num_output_units,
                use_bias= self._use_bias,
                name= "output"
                )

            new_State = self._output_state_activation(output)

        return tf.concat([hidden, output], axis=1), new_State