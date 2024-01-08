class MHA(nn.Module):
  num_hiddens: int
  num_heads: int
  bias: bool = False

  def setup(self):
    self.W_q = nn.Dense(self.num_hiddens*self.num_heads, use_bias=self.bias)
    self.W_k = nn.Dense(self.num_hiddens*self.num_heads, use_bias=self.bias)
    self.W_v = nn.Dense(self.num_hiddens*self.num_heads, use_bias=self.bias)
    self.W_o = nn.Dense(self.num_hiddens, use_bias=self.bias)

    # vectorize along head dimension
    # self.mha_dp_atn = jax.vmap(self.dotprod_atn, (1, 1, 1), 1)
    self.batch_softmax = jax.vmap(self.softmax, (0,), 0)

  def softmax(self, z):
    """
    Computes softmax along every row in a 2D matrix
    """
    zmax = jnp.max(z, axis=1)
    z = jnp.exp(z - zmax)
    scores = z / jnp.sum(z, axis=1)

  @nn.compact
  def __call__(self, Q, K, V):

    # Batch, time, embedding dimension
    B,T,D = Q.shape

    # compute QKV for each head in parallel
    Q = self.W_q(Q).reshape((B, self.num_heads, T, self.num_hiddens))
    K = self.W_k(K).reshape((B, self.num_heads, self.num_hiddens, T))
    V = self.W_v(V).reshape((B, self.num_heads, T, self.num_hiddens))

    sim = Q @ K
    c = self.batch_softmax(jnp.ones((B, T, T)))

    # reweigh V for each head, flatten

    return c

mha = MHA(12, 2)

# dummy input with batch, time, embedding dimension
X_ex = jnp.ones((BATCH_SIZE, CTX_LEN, 3))

# retrieve model state WRT self attention
# initialize parameters to dummy input shape for testing
params = mha.init(jax_key, X_ex, X_ex, X_ex)
print(mha.apply(params, X_ex, X_ex, X_ex).shape)
print(mha.tabulate(jax_key, X_ex, X_ex, X_ex))
