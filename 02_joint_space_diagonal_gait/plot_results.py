"""Plot CSV files saved by joint_space_diagonal_gait.py --headless.

Usage: python plot_results.py [file.csv ...]
With no arguments it reads every results/freq_*.csv.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LEGS = ('FL', 'FR', 'RL', 'RR')
WINDOW = (5., 10.)   # s, full gait amplitude, used for the numbers in the table
COLORS = ['#0072B2', '#009E73', '#D55E00', '#CC79A7']


def freq_of(path):
    return float(Path(path).stem.split('_')[1])


def joint_rmse(d):
    """RMS of (actual - target) over the 12 joints at each logged time."""
    actual = [n for n in d.dtype.names if n.endswith('_actual_rad')]
    err = np.stack([d[n] - d[n.replace('_actual_', '_target_')] for n in actual])
    return np.sqrt((err**2).mean(axis=0))


def window_stats(d):
    t = d['time_s']
    inside = (t >= WINDOW[0]) & (t <= WINDOW[1])
    forward = np.interp(WINDOW[1], t, d['x_m']) - np.interp(WINDOW[0], t, d['x_m'])
    return {
        'forward_m': forward,
        'speed_mps': forward / (WINDOW[1] - WINDOW[0]),
        'tilt_deg': d['tilt_deg'][inside].max(),
        'torque_Nm': d['max_torque_Nm'][inside].max(),
        'rmse_rad': np.sqrt((joint_rmse(d)[inside]**2).mean()),
    }


def contact_stats(d):
    """Share of the window each foot is off the ground, and how often a diagonal pair is up together."""
    t = d['time_s']
    inside = (t >= WINDOW[0]) & (t <= WINDOW[1])
    up = {leg: d[f'{leg}_contact'][inside] == 0 for leg in LEGS}

    def together(a, b):   # both up, as a share of the time at least one of them is up
        either = (up[a] | up[b]).mean()
        return 100*(up[a] & up[b]).mean() / either if either else 0.

    all_up = 100*np.all([up[leg] for leg in LEGS], axis=0).mean()
    return {leg: 100*up[leg].mean() for leg in LEGS}, together('FL', 'RR'), together('FR', 'RL'), all_up


def main():
    paths = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / 'results').glob('freq_*.csv'), key=freq_of)
    if not paths:
        sys.exit('No CSV files found. Run joint_space_diagonal_gait.py --headless first.')
    runs = [(freq_of(p), np.genfromtxt(p, delimiter=',', names=True)) for p in paths]
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)

    print(f'Window {WINDOW[0]:g}-{WINDOW[1]:g} s')
    print('freq_Hz  forward_m  speed_m/s  max_tilt_deg  max_torque_Nm  joint_rmse_rad')
    for f, d in runs:
        s = window_stats(d)
        print(f'{f:7g}  {s["forward_m"]:9.4f}  {s["speed_mps"]:9.4f}  {s["tilt_deg"]:12.2f}  '
              f'{s["torque_Nm"]:13.1f}  {s["rmse_rad"]:14.4f}')

    print('\nFoot contact in the same window (contact is logged every 20 ms)')
    print('freq_Hz  off-ground %: FL    FR    RL    RR   FL+RR up together %  FR+RL up together %  all four up %')
    for f, d in runs:
        air, pair_a, pair_b, all_up = contact_stats(d)
        print(f'{f:7g}  {"":13}{air["FL"]:5.1f} {air["FR"]:5.1f} {air["RL"]:5.1f} {air["RR"]:5.1f}'
              f'  {pair_a:19.1f}  {pair_b:19.1f}  {all_up:13.1f}')

    print('\nForward position over the whole run (m)')
    print('freq_Hz  lowest  at_time_s  final')
    for f, d in runs:
        i = d['x_m'].argmin()
        print(f'{f:7g}  {d["x_m"][i]:6.3f}  {d["time_s"][i]:9.2f}  {d["x_m"][-1]:5.3f}')

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        (ax[0, 0], lambda d: d['x_m'], 'Forward position (m)'),
        (ax[0, 1], lambda d: d['tilt_deg'], 'Body tilt (deg)'),
        (ax[1, 0], joint_rmse, 'Joint tracking RMSE (rad)'),
        (ax[1, 1], lambda d: d['max_torque_Nm'], 'Largest motor torque (N m)'),
    ]
    for a, get, label in panels:
        for (f, d), c in zip(runs, COLORS):
            a.plot(d['time_s'], get(d), color=c, lw=1.2, label=f'{f:g} Hz')
        a.axvspan(*WINDOW, color='0.9', zorder=0)
        a.set_ylabel(label)
        a.grid(alpha=0.3)
    for a in ax[1]:
        a.set_xlabel('Simulation time (s)')
    ax[0, 0].legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out / 'gait_comparison.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.2*len(runs) + 0.6), sharex=True, squeeze=False)
    for a, (f, d) in zip(axes[:, 0], runs):
        flags = np.vstack([d[f'{leg}_contact'] for leg in LEGS])
        a.imshow(flags, aspect='auto', cmap='Blues', vmin=0, vmax=1, interpolation='nearest',
                 extent=[d['time_s'][0], d['time_s'][-1], len(LEGS), 0])
        a.set_yticks(np.arange(len(LEGS)) + 0.5, LEGS)
        for edge in WINDOW:
            a.axvline(edge, color='#D55E00', ls='--', lw=1)
        a.set_title(f'{f:g} Hz', loc='left', fontsize=10)
    axes[-1, 0].set_xlabel('Simulation time (s)')
    fig.suptitle('Foot contact (dark = touching a surface)')
    fig.tight_layout()
    fig.savefig(out / 'ground_contact.png', dpi=150)
    plt.close(fig)
    print(f'saved figures to {out}')


if __name__ == '__main__':
    main()
