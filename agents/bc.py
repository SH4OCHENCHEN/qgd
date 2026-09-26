from typing import Any
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor


class BCAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params):
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -log_prob.mean()
        actions = dist.mode()
        mse = jnp.square(actions - batch['actions']).sum(axis=-1)
        if self.config['tanh_squash']:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev()
        return (
            actor_loss,
            {
                'actor_loss': actor_loss,
                'negative_log_likelihood': actor_loss,
                'log_prob': log_prob.mean(),
                'mse': mse.mean(),
                'std': action_std.mean(),
            },
        )

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        del rng
        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        info = {f'actor/{k}': v for k, v in actor_info.items()}
        return (actor_loss, info)

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return (self.replace(network=new_network, rng=new_rng), info)

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        del seed
        dist = self.network.select('actor')(observations, temperature=temperature)
        actions = dist.mode()
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        action_dim = ex_actions.shape[-1]
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = encoder_module()
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=False,
            const_std=True,
            final_fc_init_scale=config['actor_fc_scale'],
            encoder=encoders.get('actor'),
        )
        network_info = dict(actor=(actor_def, (ex_observations,)))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='bc',
            lr=0.0003,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            tanh_squash=True,
            actor_fc_scale=0.01,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
