"""Go2 foot trajectories in the body frame -> analytical IK -> joint PD. Simulation only."""
import argparse
import math
import threading
import time
from pathlib import Path

import glfw  # installed with mujoco; used only for key codes
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
SCENE = ROOT.parent / 'models' / 'unitree_go2' / 'scene.xml'
LEGS = ('FL', 'FR', 'RL', 'RR')
PARTS = ('hip', 'thigh', 'calf')
DURATION = 14.   # s
KP = np.tile([80., 200., 200.], 4)   # hip, thigh, calf gains, repeated for each leg
KD = np.tile([4., 6., 6.], 4)


def smooth(u):
    """Quintic smoothstep from 0 to 1 with zero slope at both ends."""
    u = np.clip(u, 0., 1.)
    return u**3 * (10 - 15*u + 6*u*u)


def envelope(t):
    """0 = stand, 1 = full gait: ramps up over 2-4 s, down over 10-12 s."""
    return float(smooth((t - 2) / 2) * (1 - smooth((t - 10) / 2)))


class LegIK:
    """Analytical IK for one leg: foot position in the body frame -> hip, thigh, calf angles."""

    def __init__(self, m, leg):
        self.origin = m.body_pos[m.body(leg + '_hip').id].copy()
        self.lateral = float(m.body_pos[m.body(leg + '_thigh').id, 1])
        self.l1 = -float(m.body_pos[m.body(leg + '_calf').id, 2])   # thigh length
        offset = m.geom_pos[m.geom(leg).id]                         # foot sphere centre in the calf frame
        self.l2 = float(np.hypot(offset[0], offset[2]))             # calf length to the foot centre
        self.delta = float(np.arctan2(-offset[0], -offset[2]))      # calf angle offset of that line

    def solve(self, foot):
        x, y, z = np.asarray(foot) - self.origin
        zp = -math.sqrt(y*y + z*z - self.lateral*self.lateral)      # height in the leg plane
        hip = math.atan2(z, y) - math.atan2(zp, self.lateral)
        hip = (hip + math.pi) % (2*math.pi) - math.pi
        c = (x*x + zp*zp - self.l1*self.l1 - self.l2*self.l2) / (2*self.l1*self.l2)
        knee = -math.acos(float(np.clip(c, -1, 1)))
        thigh = math.atan2(-x, -zp) - math.atan2(self.l2*math.sin(knee), self.l1 + self.l2*math.cos(knee))
        return np.array([hip, thigh, knee - self.delta])


class Experiment:
    def __init__(self, freq=1.2, step_length=.05, step_height=.04, duty=.65):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.freq, self.step_length, self.step_height, self.duty = freq, step_length, step_height, duty
        self.act = np.array([self.m.actuator(f'{leg}_{p}').id for leg in LEGS for p in PARTS])
        joints = self.m.actuator_trnid[self.act, 0]
        self.qadr, self.vadr = self.m.jnt_qposadr[joints], self.m.jnt_dofadr[joints]
        self.feet = np.array([self.m.geom(leg).id for leg in LEGS])
        self.base = self.m.body('base').id
        self.ik = [LegIK(self.m, leg) for leg in LEGS]
        self.reset()

    def body_feet(self):
        """Foot sphere centres in the body frame, shape (4, 3)."""
        d = self.d
        return (d.geom_xpos[self.feet] - d.xpos[self.base]) @ d.xmat[self.base].reshape(3, 3)

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key('home').id)
        mujoco.mj_forward(self.m, self.d)
        self.nominal = self.body_feet().copy()
        self.errors = []   # per-foot tracking error samples between 5 and 10 s
        self.max_tilt = 0.
        self.max_torque = 0.

    def foot_reference(self, t):
        """Target foot centres in the body frame (+X is forward)."""
        phase = (self.freq*(t - 2) + np.array([0., .5, .5, 0.])) % 1   # FL and RR together, FR and RL half a cycle later
        feet = self.nominal.copy()
        for i, p in enumerate(phase):
            if p < self.duty:   # stance: foot moves backwards on the ground
                u = p / self.duty
                dx, dz = self.step_length*(.5 - smooth(u)), 0.
            else:               # swing: foot moves forwards and lifts
                u = (p - self.duty) / (1 - self.duty)
                dx = self.step_length*(-.5 + smooth(u))
                dz = self.step_height*math.sin(math.pi*u)**2
            feet[i] += [envelope(t)*dx, 0., envelope(t)*dz]
        return feet

    def reference(self, t):
        """Foot targets and the 12 joint angles that give them."""
        feet = self.foot_reference(t)
        return np.concatenate([ik.solve(f) for ik, f in zip(self.ik, feet)]), feet

    def step(self):
        if self.d.time >= DURATION:
            return
        t, h = self.d.time, 1e-4
        q, _ = self.reference(t)
        qdot = (self.reference(t + h)[0] - self.reference(t - h)[0]) / (2*h)
        # qfrc_bias is gravity and Coriolis feedforward in joint space, not body balance
        tau = (KP*(q - self.d.qpos[self.qadr]) + KD*(qdot - self.d.qvel[self.vadr])
               + self.d.qfrc_bias[self.vadr])
        self.d.ctrl[self.act] = tau   # MuJoCo clamps it to the motor ctrlrange from go2.xml
        mujoco.mj_step(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        _, feet = self.reference(self.d.time)
        tilt = math.degrees(math.acos(float(np.clip(self.d.xmat[self.base, 8], -1, 1))))
        self.max_tilt = max(self.max_tilt, tilt)
        self.max_torque = max(self.max_torque, float(np.abs(self.d.actuator_force).max()))
        if 5. < self.d.time <= 10.:
            self.errors.append(np.linalg.norm(self.body_feet() - feet, axis=1))

    def summary(self):
        rmse = np.sqrt(np.mean(np.array(self.errors)**2))
        return (f'forward {self.d.qpos[0]:.3f} m | lateral {self.d.qpos[1]:.3f} m | '
                f'foot tracking RMSE (5-10 s) {rmse*1000:.1f} mm | '
                f'max tilt {self.max_tilt:.1f} deg | max torque {self.max_torque:.1f} N m')


def run_viewer(e):
    from mujoco import viewer
    replay, toggle = threading.Event(), threading.Event()

    def on_key(key):
        if key in (ord('R'), glfw.KEY_F9):
            replay.set()
        if key in (glfw.KEY_SPACE, glfw.KEY_F10):
            toggle.set()

    paused = False
    with viewer.launch_passive(e.m, e.d, key_callback=on_key) as v:
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.5, 135, -20
        while v.is_running():
            tick = time.perf_counter()
            if replay.is_set():
                with v.lock():
                    e.reset()
                replay.clear()
                paused = False
            if toggle.is_set():
                paused = not paused
                toggle.clear()
            with v.lock():
                if not paused:
                    for _ in range(round(.02 / e.m.opt.timestep)):
                        e.step()
                v.cam.lookat[:] = e.d.qpos[:3]
            label = f'{e.d.time:.1f} / {DURATION:.0f} s' + ('  paused' if paused else '')
            v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         label + '\nR / F9: replay   Space / F10: pause', ''))
            v.sync()
            time.sleep(max(0., .02 - (time.perf_counter() - tick)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='no window; print a summary')
    args = parser.parse_args()
    e = Experiment()
    if args.headless:
        while e.d.time < DURATION:
            e.step()
        print(e.summary())
    else:
        run_viewer(e)


if __name__ == '__main__':
    main()
