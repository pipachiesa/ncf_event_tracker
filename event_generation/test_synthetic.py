"""
End-to-end smoke test for the event generator using a hand-built synthetic
Metrica-format tracking clip. No external data required.

The scripted clip contains, in order:
    KICK OFF -> PASS -> PASS -> (intercepted) -> BALL LOST / RECOVERY ->
    ball cleared out for a THROW IN -> PASS -> SHOT -> GOAL -> KICK OFF.

Run from the repository root:  python event_generation/test_synthetic.py
"""

import os
import sys

# Make the shared tracking/event library importable.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_cleanup"))

from lib.match import Match  # noqa: E402

# Players: id, team, name, base (x, y) in normalised [0, 1] pitch coordinates.
# Home attacks the right goal (x = 1); Away attacks the left goal (x = 0).
PLAYERS = [
    ("1", "Home", "H-GK", 0.05, 0.50),
    ("2", "Home", "H-DEF-L", 0.20, 0.35),
    ("3", "Home", "H-DEF-R", 0.20, 0.65),
    ("4", "Home", "H-MID", 0.50, 0.50),
    ("5", "Home", "H-FW-L", 0.62, 0.40),
    ("6", "Home", "H-FW-R", 0.78, 0.55),
    ("7", "Away", "A-GK", 0.95, 0.50),
    ("8", "Away", "A-DEF-L", 0.80, 0.35),
    ("9", "Away", "A-DEF-R", 0.80, 0.65),
    ("10", "Away", "A-MID", 0.83, 0.50),
]
BASE = {pid: (x, y) for pid, _, _, x, y in PLAYERS}


def lerp(a, b, t):
    return a + (b - a) * t


def flight(p0, p1, n):
    """n interpolated points strictly between p0 and p1 (ball in the air)."""
    return [(lerp(p0[0], p1[0], i / (n + 1)), lerp(p0[1], p1[1], i / (n + 1)))
            for i in range(1, n + 1)]


# Build the ball trajectory and any per-frame player overrides, frame by frame.
ball_by_frame = {}
overrides = {}  # frame -> {player_id: (x, y)}


def hold(frames, pid):
    """Player ``pid`` holds the ball at its base position for ``frames``."""
    out = []
    for _ in range(frames):
        out.append(BASE[pid])
    return out


def add(frames_points):
    start = len(ball_by_frame) + 1
    for i, pt in enumerate(frames_points):
        ball_by_frame[start + i] = pt


# 1-10  kick off, H-MID (4) on the ball at the centre spot
add([(0.50, 0.50)] * 10)
# 11-15 pass H-MID -> H-FW-L
add(flight(BASE["4"], BASE["5"], 5))
# 16-25 H-FW-L holds
add(hold(10, "5"))
# 26-30 pass H-FW-L -> H-FW-R
add(flight(BASE["5"], BASE["6"], 5))
# 31-40 H-FW-R holds
add(hold(10, "6"))
# 41-45 H-FW-R tries to play forward but A-MID intercepts
add(flight(BASE["6"], BASE["10"], 5))
# 46-55 A-MID holds (Away in possession)
add(hold(10, "10"))
# 56-58 A-MID clears it out over the bottom touchline
add([(0.70, 1.01), (0.66, 1.03), (0.62, 1.05)])
# 59-60 ball dead (out of play)
add([None, None])
# 61-70 Home throw-in; H-MID (4) steps to the touchline to take it
throw_spot = (0.60, 0.985)
start = len(ball_by_frame) + 1
for i in range(10):
    ball_by_frame[start + i] = throw_spot
    overrides[start + i] = {"4": throw_spot}
# 71-75 pass from the throw -> H-FW-R steps up to (0.88, 0.50)
shooter_spot = (0.88, 0.50)
add(flight(throw_spot, shooter_spot, 5))
# 76-85 H-FW-R holds at the edge of the box
start = len(ball_by_frame) + 1
for i in range(10):
    ball_by_frame[start + i] = shooter_spot
    overrides[start + i] = {"6": shooter_spot}
# 86-92 SHOT: ball flies into the goal and crosses the line
add([(0.92, 0.50), (0.95, 0.50), (0.98, 0.50), (1.00, 0.50),
     (1.02, 0.50), (1.04, 0.50), (1.05, 0.50)])
# 93-100 ball dead in the net
add([None] * 8)
# 101-110 Away kick off from the centre spot (A-MID takes it). H-MID is based
# on the centre spot too, so step it aside to leave A-MID clearly on the ball.
start = len(ball_by_frame) + 1
for i in range(10):
    ball_by_frame[start + i] = (0.50, 0.50)
    overrides[start + i] = {"10": (0.50, 0.50), "4": (0.30, 0.50)}

TOTAL_FRAMES = len(ball_by_frame)


def write_tracking_csv(path):
    # Metrica header: 3 rows (team / id / name), then Period, Frame, Time, pairs.
    team_row = ["", "", ""]
    id_row = ["", "", ""]
    name_row = ["Period", "Frame", "Time [s]"]
    for pid, team, name, _, _ in PLAYERS:
        team_row += [team, ""]
        id_row += [pid, ""]
        name_row += [name, ""]
    team_row += ["", ""]
    id_row += ["", ""]
    name_row += ["Ball", ""]

    lines = [",".join(map(str, r)) for r in (team_row, id_row, name_row)]
    for fn in range(1, TOTAL_FRAMES + 1):
        row = [1, fn, round(fn * 0.04, 3)]
        frame_overrides = overrides.get(fn, {})
        for pid, _, _, _, _ in PLAYERS:
            x, y = frame_overrides.get(pid, BASE[pid])
            row += [x, y]
        ball = ball_by_frame.get(fn)
        if ball is None:
            row += [0, 0]
        else:
            row += [ball[0], ball[1]]
        lines.append(",".join(map(str, row)))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "synthetic_tracking.csv")
    write_tracking_csv(csv_path)

    match = Match()
    match.import_metrica(here + os.sep, "synthetic_tracking.csv")
    print(f"Imported {match.frames} frames, {len(match.players)} players.\n")

    log = match.generate_events()

    print("Detected possessions:")
    for poss in match.event_generator.possessions:
        print(f"  {poss}")

    print("\nDetected events:")
    log.print()

    print("\nEvent summary:", log.summary())

    out = log.export(path=os.path.join(here, "output"), file_name="synthetic_events.csv")
    print(f"\nExported event data -> {out}")

    # Basic assertions so the smoke test fails loudly if logic regresses.
    types = [e.type for e in log]
    subtypes = [e.subtype for e in log]
    assert "PASS" in types, "expected at least one PASS"
    assert "BALL LOST" in types, "expected a BALL LOST (interception/turnover)"
    assert "SHOT" in types, "expected a SHOT"
    assert "GOAL" in types, "expected a GOAL"
    assert "SET PIECE" in types, "expected SET PIECE restarts"
    assert "KICK OFF" in subtypes, "expected a KICK OFF restart"
    assert "THROW IN" in subtypes, "expected a THROW IN restart"
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
