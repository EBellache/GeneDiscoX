# genedisco/active_learning_methods/acquisition_functions/disco_bax_two_stage_jax.py

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, AnyStr

from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction
from slingpy import AbstractDataSource, AbstractBaseModel

# Import the JAX-based mini-batch K-Means
from genedisco.active_learning_methods.algorithms.mini_batch_kmeans_jax import mini_batch_kmeans


#############
# BAX scoring
#############
@jax.jit
def compute_bax_scores(X_jax: jnp.ndarray, gp_model, rng_key: jax.random.PRNGKey, mc_samples: int = 20) -> jnp.ndarray:
    # sample f ~ GP posterior
    f_samples = gp_model.sample_functions(X_jax, num_samples=mc_samples, rng_key=rng_key)
    # a simple BAX score = average across draws
    scores = jnp.mean(f_samples, axis=0)
    return scores


#####################################
# DiscoBAXTwoStageJax with mini-batch K-Means
#####################################
class DiscoBAXTwoStageJax(BaseBatchAcquisitionFunction):
    """
    Two-stage DiscoBAX:
      (1) BAX scoring to find top (beta * batch_size)
      (2) JAX-based mini-batch K-Means on that top subset to ensure diversity
    """

    def __init__(self, beta: int = 10, mc_samples: int = 20, random_state: int = 42,
                 mbkm_iters: int = 10, mbkm_batch_size: int = 256):
        """
        :param beta: multiplier for Stage 1 filtering
        :param mc_samples: # of Monte Carlo draws for BAX
        :param random_state: seed
        :param mbkm_iters: # of mini-batch kmeans iterations
        :param mbkm_batch_size: size of each mini-batch in the clustering step
        """
        super().__init__()
        self.beta = beta
        self.mc_samples = mc_samples
        self.random_state = random_state
        self.mbkm_iters = mbkm_iters
        self.mbkm_batch_size = mbkm_batch_size

    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr],
                 last_model: AbstractBaseModel) -> List[AnyStr]:

        # Stage 1: BAX scoring
        candidate_data = dataset_x.subset(available_indices)
        X_np = np.array(candidate_data.get_data())  # (N, d)
        N = X_np.shape[0]
        X_jax = jax.device_put(jnp.array(X_np, dtype=jnp.float32))

        rng_key = jax.random.PRNGKey(self.random_state)
        scores_jax = compute_bax_scores(X_jax, last_model, rng_key, mc_samples=self.mc_samples)
        scores = np.array(scores_jax)  # to CPU

        topK = min(self.beta * batch_size, N)
        top_indices = np.argpartition(scores, -topK)[-topK:]
        # sort descending
        top_indices_sorted = top_indices[np.argsort(-scores[top_indices])]

        # Stage 2: JAX-based mini-batch K-Means on the top subset
        # We want to cluster topK points into 'batch_size' clusters
        # Then pick the best-scoring point from each cluster.
        X_top_np = X_np[top_indices_sorted]
        X_top_jax = jax.device_put(jnp.array(X_top_np, dtype=jnp.float32))

        # Run mini-batch K-Means
        rng_key, subkey = jax.random.split(rng_key)
        centroids, assignments = mini_batch_kmeans(
            X_top_jax,
            batch_size,  # we want 'batch_size' clusters
            subkey,
            num_iters=self.mbkm_iters,
            batch_size=self.mbkm_batch_size
        )
        # assignments: shape (topK,)

        # For each cluster c in [0..batch_size-1], pick the point with highest BAX score
        # This yields exactly 'batch_size' distinct picks, as each cluster has 1 chosen.
        final_pool_inds = []
        scores_top = scores[top_indices_sorted]  # shape (topK,)
        assignments_np = np.array(assignments)  # CPU array for indexing
        for c in range(batch_size):
            c_inds = np.where(assignments_np == c)[0]
            if len(c_inds) == 0:
                continue
            best_ind = c_inds[np.argmax(scores_top[c_inds])]
            final_pool_inds.append(best_ind)
        final_pool_inds = np.array(final_pool_inds, dtype=int)

        # map these cluster-chosen indices back to the global 'available_indices'
        chosen_global_indices = top_indices_sorted[final_pool_inds]
        selected_ids = [available_indices[i] for i in chosen_global_indices]
        return selected_ids
