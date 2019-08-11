# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A Latent Dirichlet Allocation (LDA) built on tensorflow probability.

Adapted from the example in the tensorflow probability repository. All 
written below is from the tutorial.

LDA [1] is a topic model for documents represented as bag-of-words
(word counts). It attempts to find a set of topics so that every document from
the corpus is well-described by a few topics.

Suppose that there are `V` words in the vocabulary and we want to learn `K`
topics. For each document, let `w` be its `V`-dimensional vector of word counts
and `theta` be its `K`-dimensional vector of topics. Let `Beta` be a `KxN`
matrix in which each row is a discrete distribution over words in the
corresponding topic (in other words, belong to a unit simplex). Also, let
`alpha` be the `K`-dimensional vector of prior distribution parameters
(prior topic weights).

The model we consider here is obtained from the standard LDA by collapsing
the (non-reparameterizable) Categorical distribution over the topics
[1, Sec. 3.2; 3]. Then, the prior distribution is
`p(theta) = Dirichlet(theta | alpha)`, and the likelihood is
`p(w | theta, Beta) = OneHotCategorical(w | theta Beta)`. This means that we
sample the words from a Categorical distribution that is a weighted average
of topics, with the weights specified by `theta`. The number of samples (words)
in the document is assumed to be known, and the words are sampled independently.
We follow [2] and perform amortized variational inference similarly to
Variational Autoencoders. We use a neural network encoder to
parameterize a Dirichlet variational posterior distribution `q(theta | w)`.
Then, an evidence lower bound (ELBO) is maximized with respect to
`alpha`, `Beta` and the parameters of the variational posterior distribution.

We use the preprocessed version of 20 newsgroups dataset from [3].
This implementation uses the hyperparameters of [2] and reproduces the reported
results (test perplexity ~875).

Example output for the final iteration:

```none
elbo
-567.829

loss
567.883

global_step
180000

reconstruction
-562.065

topics
index=8 alpha=0.46 write article get think one like know say go make
index=21 alpha=0.29 use get thanks one write know anyone car please like
index=0 alpha=0.09 file use key program window image available information
index=43 alpha=0.08 drive use card disk system problem windows driver mac run
index=6 alpha=0.07 god one say christian jesus believe people bible think man
index=5 alpha=0.07 space year new program use research launch university nasa
index=33 alpha=0.07 government gun law people state use right weapon crime
index=36 alpha=0.05 game team play player year win season hockey league score
index=42 alpha=0.05 go say get know come one think people see tell
index=49 alpha=0.04 bike article write post get ride dod car one go

kl
5.76408

perplexity
873.206
```

#### References

[1]: David M. Blei, Andrew Y. Ng, Michael I. Jordan. Latent Dirichlet
     Allocation. In _Journal of Machine Learning Research_, 2003.
     http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
[2]: Michael Figurnov, Shakir Mohamed, Andriy Mnih. Implicit Reparameterization
     Gradients, 2018
     https://arxiv.org/abs/1805.08498
[3]: Akash Srivastava, Charles Sutton. Autoencoding Variational Inference For
     Topic Models. In _International Conference on Learning Representations_,
     2017.
     https://arxiv.org/abs/1703.01488
"""
import tensorflow as tf
import tensorflow_probability as tfp
from os.path import join, dirname
import numpy as np
import os
import scipy

tfd = tfp.distributions


def _clip_dirichlet_parameters(x):
    """Clips the Dirichlet parameters to the numerically stable KL region."""
    return tf.clip_by_value(x, 1e-3, 1e3)


def make_encoder(activation, num_topics, layer_sizes):
    """Create the encoder function.
    Args:
        activation: Activation function to use.
        num_topics: The number of topics.
        layer_sizes: The number of hidden units per layer in the encoder.
    Returns:
        encoder: A `callable` mapping a bag-of-words `Tensor` to a
        `tfd.Distribution` instance over topics.
    """
    encoder_net = tf.keras.Sequential()
    for num_hidden_units in layer_sizes:
        encoder_net.add(
            tf.keras.layers.Dense(
                num_hidden_units,
                activation=activation,
                kernel_initializer=tf.compat.v1.glorot_normal_initializer(),
            )
        )
    encoder_net.add(
        tf.keras.layers.Dense(
            num_topics,
            activation=tf.nn.softplus,
            kernel_initializer=tf.compat.v1.glorot_normal_initializer(),
        )
    )


    def encoder(bag_of_words):
        net = _clip_dirichlet_parameters(encoder_net(bag_of_words))
        return tfd.Dirichlet(concentration=net, name="topics_posterior")

    return encoder


def make_decoder(num_topics, num_words):
    """Create the decoder function.
    Args:
        num_topics: The number of topics.
        num_words: The number of words.
    Returns:
        decoder: A `callable` mapping a `Tensor` of encodings to a
        `tfd.Distribution` instance over words.
    """
    topics_words_logits = tf.compat.v1.get_variable(
        "topics_words_logits",
        shape=[num_topics, num_words],
        initializer=tf.compat.v1.glorot_normal_initializer(),
    )
    topics_words = tf.nn.softmax(topics_words_logits, axis=-1)

    def decoder(topics):
        word_probs = tf.matmul(topics, topics_words)
        # The observations are bag of words and therefore not one-hot. However,
        # log_prob of OneHotCategorical computes the probability correctly in
        # this case.
        return tfd.OneHotCategorical(probs=word_probs, name="bag_of_words")

    return decoder, topics_words


def make_prior(num_topics, initial_value):
    """Create the prior distribution.
    Args:
      num_topics: Number of topics.
      initial_value: The starting value for the prior parameters.
    Returns:
      prior: A `callable` that returns a `tf.distribution.Distribution`
          instance, the prior distribution.
      prior_variables: A `list` of `Variable` objects, the trainable parameters
          of the prior.
    """

    def _softplus_inverse(x):
        return np.log(np.expm1(x))

    logit_concentration = tf.compat.v1.get_variable(
        "logit_concentration",
        shape=[1, num_topics],
        initializer=tf.compat.v1.initializers.constant(
            _softplus_inverse(initial_value)
        ),
    )
    concentration = _clip_dirichlet_parameters(tf.nn.softplus(logit_concentration))

    def prior():
        return tfd.Dirichlet(concentration=concentration, name="topics_prior")

    prior_variables = [logit_concentration]

    return prior, prior_variables


def model_fn(features, labels, mode, params, config):
    """Build the model function for use in an estimator.
    Arguments:
      features: The input features for the estimator.
      labels: The labels, unused here.
      mode: Signifies whether it is train or test or predict.
      params: Some hyperparameters as a dictionary.
      config: The RunConfig, unused here.
    Returns:
      EstimatorSpec: A tf.estimator.EstimatorSpec instance.
    """
    del labels, config

    encoder = make_encoder(
        params["activation"], params["num_topics"], params["layer_sizes"]
    )
    decoder, topics_words = make_decoder(params["num_topics"], features.shape[1])
    prior, prior_variables = make_prior(
        params["num_topics"], params["prior_initial_value"]
    )

    topics_prior = prior()
    alpha = topics_prior.concentration

    topics_posterior = encoder(features)
    topics = topics_posterior.sample()
    random_reconstruction = decoder(topics)

    reconstruction = random_reconstruction.log_prob(features)
    tf.compat.v1.summary.scalar(
        "reconstruction", tf.reduce_mean(input_tensor=reconstruction)
    )

    # Compute the KL-divergence between two Dirichlets analytically.
    # The sampled KL does not work well for "sparse" distributions
    # (see Appendix D of [2]).
    kl = tfd.kl_divergence(topics_posterior, topics_prior)
    tf.compat.v1.summary.scalar("kl", tf.reduce_mean(input_tensor=kl))

    # Ensure that the KL is non-negative (up to a very small slack).
    # Negative KL can happen due to numerical instability.
    with tf.control_dependencies(
        [tf.compat.v1.assert_greater(kl, -1e-3, message="kl")]
    ):
        kl = tf.identity(kl)

    elbo = reconstruction - kl
    avg_elbo = tf.reduce_mean(input_tensor=elbo)
    tf.compat.v1.summary.scalar("elbo", avg_elbo)
    loss = -avg_elbo

    # Perform variational inference by minimizing the -ELBO.
    global_step = tf.compat.v1.train.get_or_create_global_step()
    optimizer = tf.compat.v1.train.AdamOptimizer(params["learning_rate"])

    # This implements the "burn-in" for prior parameters (see Appendix D of [2]).
    # For the first prior_burn_in_steps steps they are fixed, and then trained
    # jointly with the other parameters.
    grads_and_vars = optimizer.compute_gradients(loss)
    grads_and_vars_except_prior = [
        x for x in grads_and_vars if x[1] not in prior_variables
    ]

    def train_op_except_prior():
        return optimizer.apply_gradients(
            grads_and_vars_except_prior, global_step=global_step
        )

    def train_op_all():
        return optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    train_op = tf.cond(
        pred=global_step < params["prior_burn_in_steps"],
        true_fn=train_op_except_prior,
        false_fn=train_op_all,
    )

    # The perplexity is an exponent of the average negative ELBO per word.
    words_per_document = tf.reduce_sum(input_tensor=features, axis=1)
    log_perplexity = -elbo / words_per_document
    tf.compat.v1.summary.scalar(
        "perplexity", tf.exp(tf.reduce_mean(input_tensor=log_perplexity))
    )
    (log_perplexity_tensor, log_perplexity_update) = tf.compat.v1.metrics.mean(
        log_perplexity
    )
    perplexity_tensor = tf.exp(log_perplexity_tensor)

    # Obtain the topics summary. Implemented as a py_func for simplicity.
    # topics = tf.compat.v1.py_func(
    #     functools.partial(get_topics_strings, vocabulary=params["vocabulary"]),
    #     [topics_words, alpha],
    #     tf.string,
    #     stateful=False,
    # )
    # tf.compat.v1.summary.text("topics", topics)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            "elbo": tf.compat.v1.metrics.mean(elbo),
            "reconstruction": tf.compat.v1.metrics.mean(reconstruction),
            "kl": tf.compat.v1.metrics.mean(kl),
            "perplexity": (perplexity_tensor, log_perplexity_update),
            # "topics": (topics, tf.no_op()),
        },
    )


# def get_input_fn(X, batch_size):
#     # Prefetching makes training about 1.5x faster.
#     def _input_fn():
#         dataset = X.batch(batch_size).prefetch(32)
#         return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

#     return _input_fn

def input_fn(sparse_matrix, shuffle_and_repeat):
    # doc_ids, col_ids = zip(*all_docs)
    # num_documents = len(set(doc_ids))

    # # num_words = len(set(col_ids))
    # sparse_matrix = scipy.sparse.coo_matrix(
    #     (np.ones(len(all_docs)), (doc_ids, col_ids)),
    #     shape=(num_documents, num_words),
    #     dtype=np.float32,
    # )
    # sparse_matrix = sparse_matrix.tocsr()
    num_documents = sparse_matrix.shape[0]
    num_words = sparse_matrix.shape[1]
    dataset = tf.data.Dataset.range(num_documents)

    # For training, we shuffle each epoch and repeat the epochs.
    if shuffle_and_repeat:
        dataset = dataset.shuffle(num_documents).repeat()

    # Returns a single document as a dense TensorFlow tensor. The dataset is
    # stored as a sparse matrix outside of the graph.
    def get_row_py_func(idx):
        def get_row_python(idx_py):
            return np.squeeze(np.array(sparse_matrix[idx_py].todense()), axis=0)

        py_func = tf.compat.v1.py_func(
            get_row_python, [idx], tf.float32, stateful=False
        )
        py_func.set_shape((num_words,))
        return py_func

    dataset = dataset.map(get_row_py_func)
    dataset = dataset.batch(32).prefetch(32)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()


class LDA:
    """Latent Dirichlet Allocation implemented in stan."""

    def __init__(
        self,
        layer_sizes=[300, 300, 300],
        activation="relu",
        prior_initial_value=0.7,
        batch_size=32,
        max_steps=180000,
        n_topics=10,
        alpha=0.1,
        beta=1,
        learning_rate=3e-4,
        prior_burn_in_steps=120000,
        chains=4,
        delete_existing=True,
        model_dir=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "lda/"),
        save_checkpoints_steps=10000,
        shuffle_and_repeat=True
    ):
        self.n_topics = n_topics
        self.alpha = (alpha,)
        self.beta = beta
        self.prior_initial_value = prior_initial_value

        self.layer_sizes = layer_sizes
        self.activation = activation

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.shuffle_and_repeat = shuffle_and_repeat
        self.prior_burn_in_steps = prior_burn_in_steps

        self.chains = chains
        self.model_dir = model_dir
        self.save_checkpoints_steps = save_checkpoints_steps
        self.delete_existing = delete_existing


    def fit(self, X, X_val=None):
        """First a sparse matrix..."""

        params = {}
        params["layer_sizes"] = self.layer_sizes
        params["num_topics"] = self.n_topics
        params["activation"] = getattr(tf.nn, self.activation)
        params["prior_initial_value"] = self.prior_initial_value
        params["learning_rate"] = self.learning_rate
        params["prior_burn_in_steps"] = self.prior_burn_in_steps

        if self.delete_existing and tf.io.gfile.exists(self.model_dir):
            tf.compat.v1.logging.warn(
                "Deleting old log directory at {}".format(self.model_dir)
            )
            tf.io.gfile.rmtree(self.model_dir)

        tf.io.gfile.makedirs(self.model_dir)

        # params["vocabulary"] = vocabulary

        estimator = tf.estimator.Estimator(
            model_fn,
            params=params,
            config=tf.estimator.RunConfig(
                model_dir=self.model_dir,
                save_checkpoints_steps=self.save_checkpoints_steps,
            ),
        )

        _input_fn = lambda: input_fn(X, self.shuffle_and_repeat)
        _input_fn_eval = lambda: input_fn(X_val if X_val is not None else X, False)
        for _ in range(self.max_steps // self.save_checkpoints_steps):
            estimator.train(_input_fn, steps=self.save_checkpoints_steps)
            eval_results = estimator.evaluate(_input_fn_eval)
            # Print the evaluation results. The keys are strings specified in
            # eval_metric_ops, and the values are NumPy scalars/arrays.
            for key, value in eval_results.items():
                print(key)
                if key == "topics":
                    # Topics description is a np.array which prints better row-by-row.
                    for s in value:
                        print(s)
                else:
                    print(str(value))
                print("")
            print("")

        return self

    def transform(self, X):
        """Computes topic probabilities for unseen documents.

        """

        pass

    def perplexity(self):
        pass

    def plot_(self,):
        pass
