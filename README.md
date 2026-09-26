# QGD

Python 3.10+; Linux with CUDA 12 for the supplied JAX dependency.

Install:

```sh
python -m pip install -r requirements.txt
```

Train:

```sh
python main.py --agent=agents/qgd.py --env_name=cube-double-play-singletask-task1-v0 --seed=0
```

Defaults are the supplied implementation settings, not task-tuned presets. Override settings with `--agent.<key>=<value>`. For online fine-tuning, set `--online_steps`. D4RL tasks additionally require the MuJoCo runtime.

Agents: bc, cdp, fbrac, fql, ifql, iql, rebrac, qam, vgf.

Outputs are stored under `exp/experiments/`. Cloud logging is disabled by default. Keep generated outputs and local environment files out of the submission.
