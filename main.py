import os
import atexit
import json
import random
import shutil
import tempfile
import time
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS
flags.DEFINE_integer('enable_wandb', 0, 'Whether to use wandb.')
flags.DEFINE_string('wandb_run_group', 'experiments', 'Run group.')
flags.DEFINE_string('wandb_mode', 'disabled', 'Wandb mode.')
flags.DEFINE_integer(
    'wandb_no_local_files', 1, 'Whether to avoid persisting wandb files under the experiment directory.'
)
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
flags.DEFINE_float(
    'xla_mem_fraction',
    None,
    'JAX GPU memory fraction. If unset, uses 0.12 for non-visual envs and 0.45 for visual envs.',
)
flags.DEFINE_string(
    'visual_encoder',
    'impala_small',
    'Default encoder to use for visual environments when the agent encoder is unset.',
)
config_flags.DEFINE_config_file('agent', 'agents/cdp.py', lock_config=False)


def _set_xla_memory_fraction(env_name, xla_mem_fraction):
    if 'XLA_PYTHON_CLIENT_MEM_FRACTION' in os.environ:
        return
    fraction = xla_mem_fraction
    if fraction is None:
        fraction = 0.45 if 'visual' in env_name else 0.12
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(fraction)


def main(_):
    _set_xla_memory_fraction(FLAGS.env_name, FLAGS.xla_mem_fraction)
    import jax
    from agents import agents
    from envs.env_utils import make_env_and_datasets
    from utils.datasets import Dataset, ReplayBuffer
    from utils.evaluation import evaluate, flatten
    from utils.flax_utils import restore_agent, save_agent

    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb_run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    if FLAGS.enable_wandb and FLAGS.wandb_mode != 'disabled':
        wandb_output_dir = FLAGS.save_dir
        if FLAGS.wandb_no_local_files:
            wandb_output_dir = tempfile.mkdtemp(prefix='wandb_tmp_')

            def _cleanup_wandb_tmp_dir(tmp_dir=wandb_output_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

            atexit.register(_cleanup_wandb_tmp_dir)
        setup_wandb(
            wandb_output_dir=wandb_output_dir,
            project='experiments',
            group=FLAGS.wandb_run_group,
            name=exp_name,
            mode=FLAGS.wandb_mode,
        )
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)
    config = FLAGS.agent
    if 'visual' in FLAGS.env_name and 'encoder' in config and (config['encoder'] is None):
        config.encoder = FLAGS.visual_encoder
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack
    )
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, (
            'Rendering is currently only supported for OGBench environments.'
        )
    if FLAGS.online_steps > 0:
        assert 'visual' not in FLAGS.env_name, (
            'Online fine-tuning is currently not supported for visual environments.'
        )
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    train_dataset = Dataset.create(**train_dataset)
    replay_buffer = None
    if FLAGS.online_steps > 0:
        if FLAGS.balanced_sampling:
            example_transition = {k: v[0] for k, v in train_dataset.items()}
            replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
        else:
            replay_buffer = ReplayBuffer.create_from_initial_dataset(
                dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
            )
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] in ['rebrac']:
                dataset.return_next_actions = True
    example_batch = train_dataset.sample(1)
    assert 'rewards' in train_dataset
    example_batch['min_reward'] = float(train_dataset['rewards'].min())
    example_batch['max_reward'] = float(train_dataset['rewards'].max())
    assert example_batch['min_reward'] <= example_batch['max_reward']
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_batch, config)
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    rng = jax.random.PRNGKey(FLAGS.seed)
    expl_metrics = dict()
    done = True
    for i in tqdm.tqdm(
        range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        if i <= FLAGS.offline_steps:
            batch = train_dataset.sample(config['batch_size'])
            if config['agent_name'] in ['rebrac']:
                agent, update_info = agent.update(batch, full_update=i % config['actor_freq'] == 0)
            else:
                agent, update_info = agent.update(batch)
        else:
            if i == FLAGS.offline_steps + 1 and hasattr(agent, 'switch_config_to_online'):
                agent = agent.switch_config_to_online()
            rng, expl_rng = jax.random.split(rng)
            if done:
                obs, _ = env.reset()
            action = agent.sample_actions(observations=obs, temperature=1, seed=expl_rng)
            action = np.array(action)
            next_obs, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated
            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                reward = reward - 1.0
            replay_buffer.add_transition(
                dict(
                    observations=obs,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_obs,
                )
            )
            obs = next_obs
            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}
            if FLAGS.balanced_sampling:
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {
                    k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch
                }
            else:
                batch = replay_buffer.sample(config['batch_size'])
            batch = batch
            if config['agent_name'] in ['rebrac']:
                agent, update_info = agent.update(batch, full_update=i % config['actor_freq'] == 0)
            else:
                agent, update_info = agent.update(batch)
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            if FLAGS.enable_wandb:
                wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            eval_metrics = {}
            eval_kwargs = dict()
            eval_info, _, renders = evaluate(
                agent=agent,
                env=eval_env,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                **eval_kwargs,
            )
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v
            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video
            if FLAGS.enable_wandb:
                wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)
    train_logger.close()
    eval_logger.close()
    if FLAGS.enable_wandb and FLAGS.wandb_mode != 'disabled':
        wandb.finish()


if __name__ == '__main__':
    app.run(main)
