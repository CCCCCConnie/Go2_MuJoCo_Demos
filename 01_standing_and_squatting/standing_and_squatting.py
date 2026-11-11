"""Go2 stand -> squat -> stand with joint PD control. Simulation only."""
import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parent.parent / 'models' / 'unitree_go2' / 'scene.xml'
DURATION = 12.0   # s, one full cycle
KP = np.tile([60., 80., 80.], 4)   # hip, thigh, calf gains, repeated for each leg
KD = np.tile([3., 4., 4.], 4)


def smoothstep(u):
    return u**3 * (10 - 15*u + 6*u*u)


def target(t):
    """Joint targets: stand 2 s, squat down 3 s, hold 2 s, stand up 3 s."""
    if t < 2:
        blend = 0.
    elif t < 5:
        blend = smoothstep((t - 2) / 3)
    elif t < 7:
        blend = 1.
    elif t < 10:
        blend = 1 - smoothstep((t - 7) / 3)
    else:
        blend = 0.
    return np.tile([0., .9 + .2*blend, -1.8 - .4*blend], 4)


def phase_name(t):
    return ('stand' if t < 2 else 'squat down' if t < 5 else 'hold squat' if t < 7
            else 'stand up' if t < 10 else 'stand complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['inspect', 'motion'], default='motion',
                        help='inspect shows the frozen home pose in the viewer')
    parser.add_argument('--headless', action='store_true', help='run the motion without a window')
    args = parser.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)
    d.ctrl[:] = 0   # the keyframe's ctrl values are not torque commands
    mujoco.mj_forward(m, d)
    joints = m.actuator_trnid[:, 0]
    qadr, vadr = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    base = m.body('base').id
    peak_tilt = 0.
    phase = None

    def advance():
        nonlocal peak_tilt, phase
        # PD torque; MuJoCo clamps it to the motor ctrlrange from go2.xml
        d.ctrl[:] = KP*(target(d.time) - d.qpos[qadr]) - KD*d.qvel[vadr]
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        peak_tilt = max(peak_tilt, math.degrees(math.acos(float(np.clip(d.xmat[base, 8], -1, 1)))))
        label = phase_name(d.time)
        if label != phase:
            print(f'{d.time:5.2f}s {label}', flush=True)
            phase = label

    if args.headless:
        while d.time < DURATION:
            advance()
    else:
        from mujoco import viewer
        with viewer.launch_passive(m, d) as v:
            v.cam.lookat[:] = [0, 0, .2]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.35, 135, -20
            print('One 12 s cycle, then the robot holds the stand. Close the window to exit.', flush=True)
            while v.is_running():
                tick = time.perf_counter()
                if args.mode == 'motion':
                    with v.lock():
                        advance()
                v.sync()
                step = m.opt.timestep if args.mode == 'motion' else .02
                time.sleep(max(0., step - (time.perf_counter() - tick)))
    if args.mode == 'motion' or args.headless:
        print(f'Finished at {d.time:.2f} s; peak body tilt {peak_tilt:.2f} deg')


if __name__ == '__main__':
    main()
