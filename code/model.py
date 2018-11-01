import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

class lstm_model():
    """
        Using bidirectional LSTM to learn sentence embedding 
        for users' questions
    """

    def __init__(self, FLAGS, initializer=
            tf.contrib.layers.xavier_initializer()):
        """
            lstm class initialization
        """
        # configurations
        self.hidden_size = FLAGS.hidden_size
        self.vocab_size = FLAGS.vocab_size
        self.word_emb_size = FLAGS.word_emb_size
        self.batch_size = FLAGS.batch_size
        self.learning_rate = FLAGS.learning_rate
        self.initializer = initializer
        self.s_cnum = FLAGS.s_cnum
        self.margin = FLAGS.margin
        self.keep_prob = FLAGS.keep_prob
        self.num_routing = FLAGS.num_routing
        self.output_atoms = FLAGS.output_atoms

        # parameters for self attention
        self.n = FLAGS.max_time
        self.d = FLAGS.word_emb_size
        self.d_a = FLAGS.d_a
        self.u = FLAGS.hidden_size
        self.r = FLAGS.r
        self.alpha = FLAGS.alpha

        # input data
        self.input_x = tf.placeholder("int64", [None, self.n])
        self.s_len = tf.placeholder("int64", [None])
        self.IND = tf.placeholder(tf.float32, shape=[None, self.s_cnum])

        # graph
        self.instantiate_weights()
        self.attention, self.sentence_embedding = self.inference()
        # capsule
        self.activation, self.weights_c, self.votes, self.weights_b = self.capsule()
        self.logits = self.get_logits()
        self.loss_val = self.loss()
        self.train_op = self.train()

    def instantiate_weights(self):
        """
            Initializer variable weights
        """
        with tf.name_scope("embedding"): # embedding matrix
            self.Embedding = tf.get_variable("Embedding",
                    shape=[self.vocab_size, self.word_emb_size],
                    initializer=self.initializer, trainable=False)
        with tf.name_scope("capsule_weights"):
            self.capsule_weights = tf.get_variable("capsule_weights",
                    shape=[self.r, self.hidden_size * 2,
                        self.s_cnum * self.output_atoms], initializer=self.initializer)

        # Declare trainable variables for self attention
	with tf.name_scope("self_attention_w_s1"):
            # shape(W_s1) = d_a * 2u
            self.W_s1 = tf.get_variable('W_s1', shape=[self.d_a, 2*self.u],
                initializer=self.initializer)
            # shape(W_s2) = r * d_a
            self.W_s2 = tf.get_variable('W_s2', shape=[self.r, self.d_a],
                initializer=self.initializer)

    def inference(self):
        """
            self attention
        """
        #shape:[None, sentence_length, embed_size]
        input_embed = tf.nn.embedding_lookup(
                self.Embedding, self.input_x, max_norm=1)

        cell_fw = tf.contrib.rnn.LSTMCell(self.u)
        cell_bw = tf.contrib.rnn.LSTMCell(self.u)

        H, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            input_embed,
            self.s_len,
            dtype=tf.float32)
        H = tf.concat([H[0], H[1]], axis=2)

        A = tf.nn.softmax(
            tf.map_fn(
              lambda x: tf.matmul(self.W_s2, x),
              tf.tanh(
                tf.map_fn(
                  lambda x: tf.matmul(self.W_s1, tf.transpose(x)),
                  H))))

        M = tf.matmul(A, H)
        return A, M

    def _squash(self, input_tensor):
	"""Applies norm nonlinearity (squash) to a capsule layer.
        Args:
            input_tensor: Input tensor. Shape is [batch, num_channels, num_atoms] for a
              fully connected capsule layer or
              [batch, num_channels, num_atoms, height, width] for a convolutional
              capsule layer.
        Returns:
            A tensor with same shape as input (rank 3) for output of this layer.
        """

        with tf.name_scope('norm_non_linearity'):
            norm = tf.norm(input_tensor, axis=2, keep_dims=True)
            norm_squared = norm * norm
            # 0.5 => 1
            return (input_tensor / norm) * (norm_squared / (0.5 + norm_squared))

    def _update_routing(self, votes, logit_shape, num_dims, input_dim, output_dim):
        """Sums over scaled votes and applies squash to compute the activations.
        Iteratively updates routing logits (scales) based on the similarity between
        the activation of this layer and the votes of the layer below.
        Args:
          votes: tensor, The transformed outputs of the layer below.
          biases: tensor, Bias variable.
          logit_shape: tensor, shape of the logit to be initialized.
          num_dims: scalar, number of dimmensions in votes. For fully connected
          capsule it is 4, for convolutional 6.
          input_dim: scalar, number of capsules in the input layer.
          output_dim: scalar, number of capsules in the output layer.
          num_routing: scalar, Number of routing iterations.
          leaky: boolean, if set use leaky routing.
        Returns:
          The activation tensor of the output layer after num_routing iterations.
        """

        votes_t_shape = [3, 0, 1, 2]
        for i in range(num_dims - 4):
            votes_t_shape += [i + 4]
        r_t_shape = [1, 2, 3, 0]
        for i in range(num_dims - 4):
            r_t_shape += [i + 4]
        votes_trans = tf.transpose(votes, votes_t_shape)

        def _body(i, logits, activations, route):
            """Routing while loop."""
            # route: [batch, input_dim, output_dim, ...]
            route = tf.nn.softmax(logits, dim=2)
            preactivate_unrolled = route * votes_trans
            preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)

            # delete bias to fit for unseen classes
            preactivate = tf.reduce_sum(preact_trans, axis=1)
            #preactivate = tf.reduce_sum(preact_trans, axis=1) + biases

            activation = self._squash(preactivate)
            activations = activations.write(i, activation)
            # distances: [batch, input_dim, output_dim]
            act_3d = tf.expand_dims(activation, 1)
            tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
            tile_shape[1] = input_dim
            act_replicated = tf.tile(act_3d, tile_shape)
            distances = tf.reduce_sum(votes * act_replicated, axis=3)
            logits += distances
            return (i + 1, logits, activations, route)

        activations = tf.TensorArray(
            dtype=tf.float32, size=self.num_routing, clear_after_read=False)
        logits = tf.fill(logit_shape, 0.0)
        i = tf.constant(0, dtype=tf.int32)
        route = tf.nn.softmax(logits, dim=2)
        _, logits, activations, route = tf.while_loop(
            lambda i, logits, activations, route: i < self.num_routing,
            _body,
            loop_vars=[i, logits, activations, route],
            swap_memory=True)

        return activations.read(self.num_routing - 1), logits, route

    def capsule(self):
        # input : lstm output 
        # shape: [batch_size, input_dim, input_atoms]
        #        [batch_size, 2,         hidden_size]
        # output_dim: s_cnum(34)
        # output_atoms: self.output_atoms
        input_dim = self.r
        input_atoms = self.hidden_size * 2
        output_dim = self.s_cnum
        output_atoms = self.output_atoms

        dropout_emb = tf.nn.dropout(self.sentence_embedding, self.keep_prob)
        input_tiled = tf.tile(tf.expand_dims(dropout_emb, -1),
            [1, 1, 1, output_dim * output_atoms])
        votes = tf.reduce_sum(input_tiled * self.capsule_weights, axis=2)
        votes_reshaped = tf.reshape(votes,
            [-1, input_dim, output_dim, output_atoms])

        input_shape = tf.shape(self.sentence_embedding)
        logit_shape = tf.stack([input_shape[0], input_dim, output_dim])
        activations, weights_b, weights_c = self._update_routing(
            votes=votes_reshaped,
            logit_shape=logit_shape,
            num_dims=4,
            input_dim=input_dim,
            output_dim=output_dim)

	return activations, weights_c, votes_reshaped, weights_b

    def get_logits(self):
        logits = tf.norm(self.activation, axis=-1)
        return logits

    def _margin_loss(self, labels, raw_logits, margin=0.4, downweight=0.5):
        """Penalizes deviations from margin for each logit.
        Each wrong logit costs its distance to margin. For negative logits margin is
        0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
        margin is 0.4 from each side.
        Args:
            labels: tensor, one hot encoding of ground truth.
            raw_logits: tensor, model predictions in range [0, 1]
            margin: scalar, the margin after subtracting 0.5 from raw_logits.
            downweight: scalar, the factor for negative cost.
        Returns:
            A tensor with cost for each data point of shape [batch_size].
        """
        logits = raw_logits - 0.5
        positive_cost = labels * tf.cast(tf.less(logits, margin),
            tf.float32) * tf.pow(logits - margin, 2)
        negative_cost = (1 - labels) * tf.cast(
        tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
        return 0.5 * positive_cost + downweight * 0.5 * negative_cost

    def loss(self):
	loss_val = self._margin_loss(self.IND, self.logits)
        loss_val = tf.reduce_mean(loss_val)
        self_atten_mul = tf.matmul(self.attention, tf.transpose(self.attention, perm=[0, 2, 1]))
        sample_num, att_matrix_size, _ = self_atten_mul.get_shape()
        self_atten_loss = tf.square(tf.norm(self_atten_mul - np.identity(att_matrix_size.value)))
	return 1000 * loss_val + self.alpha * tf.reduce_mean(self_atten_loss)

    def train(self):
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)
	return train_op
