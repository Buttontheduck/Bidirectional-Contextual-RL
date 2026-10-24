# Python environments

The exact package versions of every Python environment the experiments ran in, exported on 2026-10-24 from the training server (Ubuntu Linux, CPU only, AMD Ryzen 9 9950X). Use them to rebuild an environment, or to check which version produced a result.

| File | Environment | What ran in it |
|---|---|---|
| `carlos_conda_environment.yml`, `carlos_pip_freeze.txt` | `carlos` (conda, Python 3.11) | Everything in the main project: BatteryPlane, Contextual PPO, plain PPO, RecurrentPPO, the encoders, reports and figures |
| `venv_carlos-myo_pip_freeze.txt` | `carlos-myo` | `contextual_environments/myo/` (MyoSuite 3.0.0, MuJoCo 3.16.0) |
| `venv_citylearn_pip_freeze.txt` | `citylearn` | CityLearn 3.0.2 with the versions it pins |
| `venv_carlos-citylearn_pip_freeze.txt` | `carlos-citylearn` | `contextual_environments/city/` |
| `venv_gym4real_pip_freeze.txt` | `gym4real` | Gym4ReaL's MicroGridEnv with the versions its repository pins |
| `venv_carlos-gym4real_pip_freeze.txt` | `carlos-gym4real` | `contextual_environments/microgrid/` |
| `gym4real_commit.txt` | — | The Gym4ReaL source commit (774d5ce) and its repository |

Key versions in `carlos`: stable-baselines3 2.9.0, sb3-contrib 2.9.0, torch 2.13.0+cpu, gymnasium 0.29.1, numpy 2.4.6, xlstm 2.0.5, wandb 0.28.2.

## Rebuilding

**`carlos`** (Linux):
```bash
conda env create -f environments/carlos_conda_environment.yml
```
The file pins Linux builds, including the CPU-only torch wheel (`torch==2.13.0+cpu`). On macOS, create a Python 3.11 environment and install `carlos_pip_freeze.txt` with the `+cpu` suffix removed from torch.

**The side environments** (`contextual_environments/`). They are virtual environments, kept out of `carlos` so that it stays unchanged:
- **`carlos-myo`, `carlos-citylearn`, `carlos-gym4real`** are built on `carlos`'s Python with access to its packages: `python -m venv --system-site-packages <name>`, then `pip install` the packages that `carlos` lacks.
  - **`carlos-myo`:** MyoSuite 3.0.0 and MuJoCo 3.16.0. Run it with `MUJOCO_GL` unset.
  - **`carlos-citylearn`:** CityLearn 3.0.2 installed with `--no-deps` (its version limits would downgrade `carlos`), plus scikit-learn 1.9.1 and simplejson.
  - **`carlos-gym4real`:** pandas 2.2.3 and joblib. Gym4ReaL breaks with pandas 3.

  Their freeze files list `carlos`'s packages too, because the venvs see them.
- **`citylearn` and `gym4real`** are fully separate, with exactly the versions those projects pin: `python -m venv <name>`, then `pip install -r` its freeze file.
- **Gym4ReaL** is not a pip package. Clone https://github.com/Daveonwave/gym4ReaL.git at commit 774d5ce and put its folder on the path with a `.pth` file in the venv's `site-packages`. Run it from that folder, because its configuration paths are relative.
