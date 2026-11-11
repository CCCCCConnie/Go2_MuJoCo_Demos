# Go2 MuJoCo Motion

Three Unitree Go2 simulations in MuJoCo, tested on Windows with Python 3.11.9.

| Demo | What it does | Script |
| --- | --- | --- |
| 01 | Stand, squat down, hold, stand up with joint PD control | [standing_and_squatting.py](01_standing_and_squatting/standing_and_squatting.py) |
| 02 | Open-loop diagonal gait from sinusoidal joint targets | [joint_space_diagonal_gait.py](02_joint_space_diagonal_gait/joint_space_diagonal_gait.py) |
| 03 | Foot trajectories in the body frame, analytical IK, joint PD | [foot_trajectory_tracking.py](03_foot_trajectory_tracking/foot_trajectory_tracking.py) |

Demo 02 has its own [README](02_joint_space_diagonal_gait/README.md) with the controller, parameters and how to produce the data and figures.

## Setup

Python 3.11 on Windows. In PowerShell, from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Matplotlib is used by the plotting script for demo 02.

In VS Code, open `Go2_MuJoCo_Motion.code-workspace` and select `.venv/Scripts/python.exe` as the interpreter.

## Run

```powershell
.\.venv\Scripts\python.exe 01_standing_and_squatting/standing_and_squatting.py
.\.venv\Scripts\python.exe 02_joint_space_diagonal_gait/joint_space_diagonal_gait.py
.\.venv\Scripts\python.exe 03_foot_trajectory_tracking/foot_trajectory_tracking.py
```

Close the viewer before starting another demo. In demos 02 and 03, press `R` to replay and `Space` to pause or resume. Add `--headless` to run without a window; each demo prints a summary. Demo 02 also writes a CSV file.

## Robot model

`models/unitree_go2/` is the Unitree Go2 from [Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/unitree_go2), unchanged, under its BSD-3-Clause [license](models/unitree_go2/LICENSE).
