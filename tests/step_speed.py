import os
import sys
import time

import tqdm

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, ".."))

import easydel as ed
import easydel.trainers.training_utils as etu
import flax
import jax
from jax import numpy as jnp

from dukron import kron

dtype = jnp.float16
param_dtype = jnp.float32
sequence_length = 1024
batch_size = 4
minibatch_size = 1
loss_config = etu.LossConfig()
tx = kron(
	learning_rate=1e-4,
	b1=0.0,
	weight_decay=0.001,
	max_size_triangular=2**10,
)
config = ed.Xerxes2Config(
	vocab_size=32000,
	hidden_size=64,
	num_attention_heads=8,
	num_hidden_layers=4,
	intermediate_size=128,
	max_position_embeddings=sequence_length,
	attn_dtype=dtype,
	attn_softmax_dtype=param_dtype,
	attn_mechanism=ed.AttentionMechanisms.VANILLA,
)

model = ed.Xerxes2ForCausalLM(
	config=config,
	dtype=dtype,
	param_dtype=param_dtype,
	rngs=flax.nnx.Rngs(0),
).shard_model()
state = model.to_state()
state = state.init_tx(tx)


def loss_fn(tree, minibatch):
	module = flax.nnx.merge(state.graphdef, tree, state.graphother)
	call_batch = module.prepare_inputs_for_call(**minibatch)
	labels = call_batch.pop("labels", None)
	outputs, metrics = module.compute_loss(
		labels=labels,
		loss_config=loss_config,
		**call_batch,
	)
	return outputs.loss, metrics


batch = {
	"attention_mask": jnp.ones((batch_size, sequence_length), dtype="i4"),
	"input_ids": jnp.ones((batch_size, sequence_length), dtype="i4"),
}

gradients, metrics = etu.minibatch_call(
	state=state,
	batch=batch,
	minibatch_size=minibatch_size,
	grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
)


@jax.jit
def _call(state, gradients):
	state = etu.update_state_respectfully(
		state=state,
		gradients=gradients,
		loss_config=loss_config,
		metrics=etu.update_metrics(
			metrics=metrics,
			learning_rate_fn=lambda x: x,
			step=state.step,
			gradients=gradients,
		),
	)
	return state


times = 100
print("Compiling")
state = jax.block_until_ready(_call(state, gradients))
state = jax.block_until_ready(_call(state, gradients))
print("Compiled")

start = time.time()
for i in tqdm.tqdm(range(100)):
	state = _call(state, gradients)
took = time.time() - start

print(f"Took {took} | per iter {took / times}")
