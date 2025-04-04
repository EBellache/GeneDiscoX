# gene_disco_sparse_gp.py

import jax
import jax.numpy as jnp
import numpy as np
import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.kernels import RBF
from gpjax.mean_functions import Zero
from gpjax.likelihoods import Gaussian
import optax


class GeneDiscoSparseGP:
    """
    A GPJax-based sparse Gaussian Process model optimized for high-dimensional embeddings
    and fast Monte Carlo sampling. This model is designed for the GeneDisco challenge:
      - Uses an ARD RBF kernel to handle high-dimensional inputs.
      - Implements a variational sparse GP with inducing points.
      - Supports efficient posterior sampling for use in DiscoBAX-style acquisition.
    """

    def __init__(self, feature_dim: int, num_inducing: int = 50,
                 noise_variance: float = 1e-2,
                 rng_key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        """
        :param feature_dim: Dimensionality of input features.
        :param num_inducing: Number of inducing points.
        :param noise_variance: Initial noise variance for the Gaussian likelihood.
        :param rng_key: JAX PRNG key.
        """
        self.feature_dim = feature_dim
        self.num_inducing = num_inducing
        self.noise_variance = noise_variance
        self.rng_key = rng_key

        # Define an ARD RBF kernel (one lengthscale per input dimension)
        self.kernel = RBF(ard_dims=feature_dim)
        self.mean_function = Zero()

        # Create the GP prior
        self.prior = gpx.Prior(mean_function=self.mean_function, kernel=self.kernel)

        # Define the Gaussian likelihood.
        # (The number of datapoints is set at training time via Dataset.)
        self.likelihood = Gaussian(num_datapoints=1, noise=self.noise_variance)

        # Initialize inducing points as a (num_inducing x feature_dim) array,
        # e.g. drawn from a standard normal. In practice, you might want to use
        # a smarter initialization (e.g. k-means on training data).
        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.inducing_inputs = jax.random.normal(subkey, shape=(num_inducing, feature_dim))

        # Create the variational GP model with inducing points.
        # GPJax provides a variational model that implements a sparse inference scheme.
        self.model = gpx.VariationalGaussian(
            prior=self.prior,
            inducing_inputs=self.inducing_inputs,
            likelihood=self.likelihood,
        )

        self.params = None  # To be set upon training
        self._trained = False

    def initialize(self, train_X: np.ndarray, train_Y: np.ndarray):
        """
        Initialize the model parameters using the training data.
        """
        train_dataset = Dataset(jnp.array(train_X, dtype=jnp.float32),
                                jnp.array(train_Y, dtype=jnp.float32).reshape(-1, 1))
        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.params = self.model.initialize(subkey, train_dataset)
        return self.params

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray, num_iters: int = 200, learning_rate: float = 0.01):
        """
        Fit the sparse GP model to training data using variational inference.
        Uses GPJax’s Variational ELBO and Optax for optimization.
        """
        train_dataset = Dataset(jnp.array(train_X, dtype=jnp.float32),
                                jnp.array(train_Y, dtype=jnp.float32).reshape(-1, 1))
        self.initialize(train_X, train_Y)
        # Define the variational objective (ELBO)
        mll = gpx.VariationalELBO(self.likelihood, self.model, num_datapoints=train_Y.shape[0])
        optimizer = optax.adam(learning_rate)

        # Fit the model using GPJax’s fit routine (which is JIT-compiled)
        inference_state = gpx.fit(
            model=self.model,
            objective=mll,
            train_data=train_dataset,
            optim=optimizer,
            num_iters=num_iters,
            params=self.params,
            key=self.rng_key
        )
        self.params = inference_state.params
        self._trained = True
        return self

    def predict(self, X_new: np.ndarray):
        """
        Predict the posterior mean and standard deviation at new input points.
        Returns NumPy arrays (mean, std) of shape (N,).
        """
        if not self._trained:
            raise RuntimeError("The model must be trained before prediction.")
        X_new_jax = jnp.array(X_new, dtype=jnp.float32)
        test_dataset = Dataset(X_new_jax, None)
        predictive_dist = self.model.predict(test_data=test_dataset, params=self.params)
        mean = jnp.squeeze(predictive_dist.mean, axis=-1)
        var = jnp.squeeze(predictive_dist.variance, axis=-1)
        std = jnp.sqrt(var)
        return np.array(mean), np.array(std)

    def sample_functions(self, X_new: np.ndarray, num_samples: int = 20):
        """
        Draw Monte Carlo samples from the GP posterior at new input points.
        Returns a JAX array of shape (num_samples, N).
        This is critical for DiscoBAX-style acquisition.
        """
        if not self._trained:
            raise RuntimeError("The model must be trained before sampling.")
        X_new_jax = jnp.array(X_new, dtype=jnp.float32)
        test_dataset = Dataset(X_new_jax, None)
        predictive_dist = self.model.predict(test_data=test_dataset, params=self.params)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        samples = predictive_dist.sample(seed=subkey, sample_shape=(num_samples,))
        # samples shape: (num_samples, N, 1) – squeeze the last dimension
        return jnp.squeeze(samples, axis=-1)
