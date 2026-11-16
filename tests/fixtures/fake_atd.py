"""Writes a FAKE ATD file pair in the official FIFA/EPTS layout, for pipeline tests only.

The numbers are invented (no relation to any real match). The layout follows DATA.md and
starter/baseline.py: "<frame_id>:<x,y,s>;...;<x,y,s>:<x,y,z>", frame_id 1..N, normalised
coordinates, NaN for missing, and an XML with two <Team> elements and one <Player> per slot
(no DataFormatSpecification, so slot i = i-th <Player>).

    python tests/fixtures/fake_atd.py OUT_DIR [--frames N]

Scenario per 1000-frame block: home holds the ball (400), the ball is lost by the tracker (100),
away holds it (400), then the ball goes out over the touchline (100). Players jog except during
the out-of-play part, where they stand still.
"""

import argparse
import os

import numpy as np

N_SLOTS = 46  # 23 home + 23 away, like the competition file


def write_fake_atd(out_dir: str, n_frames: int = 199319, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    folder = os.path.join(out_dir, "tactical")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, "BAR-ATH_tactical")

    players = "\n".join(
        f'      <Player id="P{i + 1}" teamId="{"T_HOME" if i < 23 else "T_AWAY"}">'
        f"<Name>Track {i + 1}</Name><ShirtNumber>{i + 1}</ShirtNumber></Player>"
        for i in range(N_SLOTS)
    )
    with open(stem + "_atd.xml", "w", encoding="utf-8") as fh:
        fh.write(
            '<?xml version="1.0" encoding="utf-8"?>\n<main><Metadata>\n'
            "  <GlobalConfig><FrameRate>25</FrameRate></GlobalConfig>\n"
            '  <Teams><Team id="T_HOME"><Name>Home</Name></Team>'
            '<Team id="T_AWAY"><Name>Away</Name></Team></Teams>\n'
            f"  <Players>\n{players}\n  </Players>\n</Metadata></main>\n"
        )

    # 11 active slots per team; the rest are NaN (unused tracklets).
    active = list(range(11)) + list(range(23, 34))
    base = rng.uniform(0.15, 0.85, size=(N_SLOTS, 2))
    with open(stem + "_atd.txt", "w", encoding="utf-8") as fh:
        for f in range(n_frames):
            phase = f % 1000
            stopped = phase >= 900
            # ~2-4 m/s jogging while play is live, motionless while the ball is out
            jitter = np.zeros(N_SLOTS) if stopped else 0.012 * np.sin(f / 7.0 + np.arange(N_SLOTS))
            xs = base[:, 0] + jitter
            groups = []
            for s in range(N_SLOTS):
                if s in active:
                    groups.append(f"{xs[s]:.5f},{base[s, 1]:.5f},{0.0 if stopped else 2.0:.2f}")
                else:
                    groups.append("NaN,NaN,NaN")
            if phase < 400:
                ball = f"{xs[0] + 0.003:.5f},{base[0, 1]:.5f},0.00"
            elif phase < 500:
                ball = "NaN,NaN,NaN"
            elif phase < 900:
                ball = f"{xs[23] + 0.003:.5f},{base[23, 1]:.5f},0.00"
            else:
                ball = "0.50000,1.05000,0.00"  # beyond the touchline
            fh.write(f"{f + 1}:{';'.join(groups)}:{ball}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--frames", type=int, default=199319)
    args = parser.parse_args()
    write_fake_atd(args.out_dir, args.frames)
