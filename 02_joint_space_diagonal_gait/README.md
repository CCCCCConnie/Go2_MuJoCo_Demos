# 02 Joint-space diagonal gait

An open-loop gait for the Unitree Go2 in MuJoCo. Each leg follows a sinusoidal joint target, with the diagonal pairs half a cycle apart. A PD controller tracks the targets.

## Run

From the repository root, use the Python environment in the [setup instructions](../README.md#setup).

```powershell
.\.venv\Scripts\python.exe 02_joint_space_diagonal_gait/joint_space_diagonal_gait.py
```

Keys in the viewer: `R` or `F9` replays from the start, `Space` or `F10` pauses. Click the 3D view first so the window has focus.

To save data and make the figures, run each frequency headless and then plot:

```powershell
.\.venv\Scripts\python.exe 02_joint_space_diagonal_gait/joint_space_diagonal_gait.py --headless --freq 1.2
.\.venv\Scripts\python.exe 02_joint_space_diagonal_gait/joint_space_diagonal_gait.py --headless --freq 1.8
.\.venv\Scripts\python.exe 02_joint_space_diagonal_gait/joint_space_diagonal_gait.py --headless --freq 2.4
.\.venv\Scripts\python.exe 02_joint_space_diagonal_gait/plot_results.py
```

Data goes to `results/freq_<f>.csv` and figures to `figures/`. `--thigh-amp` and `--calf-amp` change the amplitudes in rad.

## Timeline

| Time (s) | Behaviour |
|---|---|
| 0–2 | stand with the PD controller |
| 2–4 | gait amplitude ramps up |
| 4–10 | full gait |
| 10–12 | amplitude ramps down |
| 12–14 | back to the standing target |

The simulation stops updating at 14 s.

## Controller

For each leg, with `q_home` taken from the model's `home` keyframe:

```
phase   = 2*pi*f*(t - 2) + offset       offset = 0 for FL and RR, pi for FR and RL
d_thigh = -A_thigh*sin(phase) + 0.5*A_calf*cos(phase)
d_calf  = -A_calf*cos(phase)
d_hip   = 0
target  = q_home + envelope(t) * d
torque  = Kp*(target - q) - Kd*dq
```

`envelope(t)` is a smoothstep that goes 0 to 1 over 2–4 s and back to 0 over 10–12 s. Gains are `Kp = [60, 80, 80]` and `Kd = [3, 4, 4]` for hip, thigh and calf. The torque is clamped by MuJoCo to the motor range in `go2.xml` (±23.7 N m for hip and thigh, ±45.43 N m for calf).

The joint targets depend on time. The PD controller uses measured joint angles and velocities to track them.

The thigh target has an extra `0.5*A_calf*cos(phase)` term. With the default amplitudes, `-A_thigh*sin(phase) + 0.5*A_calf*cos(phase)` is the same as `0.163*cos(phase + 47.5°)`. So the thigh swings with an amplitude of about 0.163 rad instead of 0.12 rad, and it is about 132° out of phase with the calf (`-A_calf*cos(phase)`).

## Parameters

| Parameter | Default | Option |
|---|---|---|
| frequency | 1.8 Hz | `--freq` |
| thigh amplitude | 0.12 rad | `--thigh-amp` |
| calf amplitude | 0.22 rad | `--calf-amp` |

The commands above vary frequency while keeping the amplitudes and gains fixed.

## Read the plots

The summary covers 5 to 10 s. The gait comparison plot shows forward position, body tilt, joint tracking error and the largest absolute actuator torque. The contact plot shows whether each foot geom was involved in a contact at each 20 ms sample.

## Model

The Go2 model is from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (BSD-3-Clause). See the [model license](../models/unitree_go2/LICENSE).
