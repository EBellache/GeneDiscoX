import jax
import jax.numpy as jnp
from jax import lax


def initialize_centroids(X, k, rng_key):
    """
    Randomly select k samples from X as initial centroids.
    X: shape (N, d)
    """
    N = X.shape[0]
    idx = jax.random.choice(rng_key, N, shape=(k,), replace=False)
    return X[idx]


def assign_clusters(X_batch, centroids):
    """
    Assign each point in the batch X_batch to its nearest centroid.
    X_batch: (batch_size, d)
    centroids: (k, d)
    Returns cluster_indices of shape (batch_size,)
    """
    distances = jnp.sum((X_batch[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    return jnp.argmin(distances, axis=1)


def update_centroids(centroids, X_batch, cluster_indices, k):
    """
    Update centroid positions given new batch assignments.
    X_batch: (batch_size, d)
    cluster_indices: (batch_size,) in [0..k-1]

    Returns updated centroids: (k, d)
    """

    def compute_new_centroid(c):
        mask = (cluster_indices == c)  # shape (batch_size,)
        count = jnp.sum(mask)
        # if no points in cluster c for this batch, do not shift centroid
        # else average the points
        # We'll do a safe update: if count=0, centroid stays the same
        old_centroid = centroids[c]
        sum_points = jnp.sum(X_batch * mask[:, None], axis=0)
        # weighted average approach: move centroid slightly toward new batch mean
        # One way is an exponential moving average or direct partial update
        # For simplicity, here we do a direct average (if count>0)
        # Weighted by alpha to avoid dramatic jumps:
        alpha = 1.0
        new_centroid = jax.lax.cond(
            count > 0,
            lambda _: old_centroid + alpha * ((sum_points / count) - old_centroid),
            lambda _: old_centroid,
            operand=None
        )
        return new_centroid

    new_centroids = jax.vmap(compute_new_centroid)(jnp.arange(k, dtype=jnp.int32))
    return new_centroids


@jax.jit
def mini_batch_kmeans(X, k, rng_key, num_iters=10, batch_size=256):
    """
    Mini-Batch K-Means using JAX.
    1) Initialize centroids
    2) Repeatedly sample a random batch from X
    3) Assign each batch point to nearest centroid, update centroids
    4) Return final centroids, plus cluster assignments for entire X

    :param X: jnp.array of shape (N, d)
    :param k: number of clusters
    :param rng_key: JAX PRNGKey
    :param num_iters: how many mini-batch passes
    :param batch_size: how large each mini-batch is

    Returns: (centroids, full_assignments)
     - centroids: shape (k, d)
     - full_assignments: shape (N,) cluster index for each point in X
    """
    N = X.shape[0]

    # 1) Initialize centroids
    rng_key, subkey = jax.random.split(rng_key)
    centroids = initialize_centroids(X, k, subkey)

    # 2) for each iteration, sample a random batch, assign, update
    def train_step(i, state):
        rng, centroids = state
        rng, sk = jax.random.split(rng)
        # sample a mini-batch of points
        batch_idx = jax.random.choice(sk, N, shape=(batch_size,), replace=False)
        X_batch = X[batch_idx]  # shape (batch_size, d)
        # assign clusters
        cluster_indices = assign_clusters(X_batch, centroids)
        # update centroids
        new_centroids = update_centroids(centroids, X_batch, cluster_indices, k)
        return (rng, new_centroids)

    init_state = (rng_key, centroids)
    final_state = lax.fori_loop(0, num_iters, train_step, init_state)
    _, final_centroids = final_state

    # 3) compute final assignments for entire dataset
    # (this can be done once to pick final representative points)
    # Might be large, so we do it in one pass if feasible
    # If X is huge, consider chunking
    assignments = assign_clusters(X, final_centroids)
    return final_centroids, assignments
