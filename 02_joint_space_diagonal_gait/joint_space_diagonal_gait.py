"""Go2 open-loop diagonal gait in joint space. Simulation only."""
import argparse
import csv
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
DURATION = 14.0   # s
LOG_EVERY = 10    # steps, so one row every 20 ms

KP = np.tile([60., 80., 80.], 4)   # hip, thigh, calf gains, repeated for each leg
KD = np.tile([3., 4., 4.], 4)


def smooth(u):
    """Quintic smoothstep from 0 to 1 with zero slope at both ends."""
    u = np.clip(u, 0., 1.)
    return u**3 * (10. - 15.*u + 6.*u**2)


class GaitSim:
    def __init__(self, freq=1.8, thigh_amp=0.12, calf_amp=0.22):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.freq, self.thigh_amp, self.calf_amp = freq, thigh_amp, calf_amp
        self.act = np.array([self.m.actuator(f'{leg}_{p}').id for leg in LEGS for p in PARTS])
        joint = self.m.actuator_trnid[self.act, 0]
        self.qadr = self.m.jnt_qposadr[joint]
        self.vadr = self.m.jnt_dofadr[joint]
        self.feet = [self.m.geom(leg).id for leg in LEGS]
        self.base = self.m.body('base').id
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key('home').id)
        mujoco.mj_forward(self.m, self.d)
        self.home = self.d.qpos[self.qadr].copy()
        self.rows = []
        self.nstep = 0

    def reference(self, t):
        """Joint targets at time t, and the gait envelope (0 = stand, 1 = full gait)."""
        envelope = float(smooth((t - 2) / 2) * (1 - smooth((t - 10) / 2)))
        # FL and RR move together, FR and RL are half a cycle behind
        phase = 2*np.pi*self.freq*(t - 2) + np.array([0., np.pi, np.pi, 0.])
        delta = np.zeros((4, 3))
        delta[:, 1] = -self.thigh_amp*np.sin(phase) + 0.5*self.calf_amp*np.cos(phase)
        delta[:, 2] = -self.calf_amp*np.cos(phase)
        return self.home + envelope*delta.ravel(), envelope

    def step(self):
        if self.d.time >= DURATION:
            return
        q_target, envelope = self.reference(self.d.time)
        # PD torque; MuJoCo clamps it to the motor ctrlrange from go2.xml
        self.d.ctrl[self.act] = KP*(q_target - self.d.qpos[self.qadr]) - KD*self.d.qvel[self.vadr]
        mujoco.mj_step(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)   # refresh contacts and body pose for logging
        self.nstep += 1
        if self.nstep % LOG_EVERY == 0:
            self.log(q_target, envelope)

    def log(self, q_target, envelope):
        d = self.d
        tilt = np.degrees(np.arccos(np.clip(d.xmat[self.base, 8], -1, 1)))
        touching = {g for c in d.contact for g in (c.geom1, c.geom2)}
        flags = [int(g in touching) for g in self.feet]
        self.rows.append([d.time, *d.qpos[:3], tilt, np.abs(d.actuator_force).max(), envelope,
                          *flags, *d.qpos[self.qadr], *q_target])

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        names = [f'{leg}_{p}' for leg in LEGS for p in PARTS]
        header = (['time_s', 'x_m', 'y_m', 'z_m', 'tilt_deg', 'max_torque_Nm', 'envelope']
                  + [f'{leg}_contact' for leg in LEGS]
                  + [f'{n}_actual_rad' for n in names]
                  + [f'{n}_target_rad' for n in names])
        with path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.rows)

    def summary(self):
        a = np.array(self.rows)
        return (f'{self.freq:g} Hz | forward {a[-1, 1]:.3f} m | lateral {a[-1, 2]:.3f} m | '
                f'max tilt {a[:, 4].max():.1f} deg | max torque {a[:, 5].max():.1f} N m')


def run_viewer(sim):
    from mujoco import viewer
    replay, toggle = threading.Event(), threading.Event()

    def on_key(key):
        if key in (ord('R'), glfw.KEY_F9):
            replay.set()
        if key in (glfw.KEY_SPACE, glfw.KEY_F10):
            toggle.set()

    paused = False
    with viewer.launch_passive(sim.m, sim.d, key_callback=on_key) as v:
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.5, 135, -20
        while v.is_running():
            tick = time.perf_counter()
            if replay.is_set():
                with v.lock():
                    sim.reset()
                replay.clear()
                paused = False
            if toggle.is_set():
                paused = not paused
                toggle.clear()
            with v.lock():
                if not paused:
                    for _ in range(round(0.02 / sim.m.opt.timestep)):
                        sim.step()
                v.cam.lookat[:] = sim.d.qpos[:3]
            label = f'{sim.d.time:.1f} / {DURATION:.0f} s' + ('  paused' if paused else '')
            v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         label + '\nR / F9: replay   Space / F10: pause', ''))
            v.sync()
            time.sleep(max(0., 0.02 - (time.perf_counter() - tick)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--headless', action='store_true', help='no window; save motion data to CSV')
    p.add_argument('--freq', type=float, default=1.8, help='gait frequency in Hz')
    p.add_argument('--thigh-amp', type=float, default=0.12, help='thigh amplitude in rad')
    p.add_argument('--calf-amp', type=float, default=0.22, help='calf amplitude in rad')
    p.add_argument('--out', type=Path, help='CSV path (default: results/freq_<freq>.csv)')
    args = p.parse_args()

    sim = GaitSim(args.freq, args.thigh_amp, args.calf_amp)
    if args.headless:
        out = args.out or ROOT / 'results' / f'freq_{args.freq:g}.csv'
        while sim.d.time < DURATION:
            sim.step()
        sim.save(out)
        print(sim.summary())
        print(f'saved {out}')
    else:
        run_viewer(sim)


if __name__ == '__main__':
    main()
