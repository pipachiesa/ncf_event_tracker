"""Player re-identification (ReID) post-process.

Standalone module that merges the fragmented player IDs that ``sv.ByteTrack``
produces (motion-only tracking mints a fresh ID every time a player is lost
and re-detected: ~187 IDs for ~25 real players on the baseline clip). It takes
the raw tracking CSV written by ``data_cleanup/main.py`` plus the source video
and writes the same CSV with ``Object ID`` rewritten to merged identities, so
event generation runs on far fewer, longer-lived players. ``main.py`` and the
tracking loop are untouched.

Two track fragments A and B are merged only if ALL five gates pass:

1. Temporally disjoint: A ends strictly before B starts.
2. Short gap: ``B.start - A.end <= tmax`` (seconds).
3. Spatially plausible: A's last position, projected forward with its exit
   velocity, lands within an allowed radius of B's first position. The radius
   is ``dmax`` plus a per-second slack for longer gaps (a player keeps moving
   while untracked).
4. Same team (majority vote over each fragment's rows; the frame-level team
   labels are noisy, the vote is not).
5. Compatible appearance: HSV histogram similarity of the two fragments'
   torso crops ``>= smin``. Colour cannot separate teammates (same shirt), so
   this gate exists to VETO merges across different-looking players, not to
   propose them.

Merging is a union-find fixpoint: merged tracks extend in time and may chain
with further fragments, so passes repeat until no merge fires. Candidate
merges are applied cheapest-first (small gap, small distance, high
similarity), re-validating every gate against the *current* merged groups, so
one fragment can never be claimed by two players. The main failure mode to
avoid is over-merging (two real players fused is worse than fragmentation),
hence every gate is a hard veto and the defaults are conservative.

Usage:
    python data_cleanup/reid.py --tracking-csv tracks.csv --video match.mp4 \
        --output merged.csv [--tmax 30] [--dmax 800] [--smin 0.30] \
        [--render check.mp4 --render-top 6]

El fps y el stride se leen del sidecar ``<tracking>.meta.json`` que escribe
main.py: definen a cuantos frames equivale ``--tmax`` y que fotograma del video
corresponde a cada frame del CSV (con ``--frame-stride`` NO son el mismo
numero). ``--fps`` explicito pisa el sidecar.

Pitch coordinates are centimetres on a 12000x7000 pitch; ``--fps`` (default
24) only converts ``--tmax`` seconds to frames. CPU only: OpenCV + numpy.
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import cv2
import numpy as np

PITCH_LENGTH_CM = 12000.0
PITCH_WIDTH_CM = 7000.0

DEFAULT_FPS = 24.0
DEFAULT_TMAX_S = 30.0     # max gap between fragments, seconds
DEFAULT_DMAX_CM = 800.0   # base spatial tolerance at gap ~ 0

# Appearance thresholds (histogram intersection, 0..1), calibrated on the
# psg_bayern baseline: same-player junction pairs have median sim ~0.39 and
# verified-good junctions sit at 0.55+, while different-team pairs have
# median ~0.11. ``smin`` is a hard colour veto applied to every junction;
# junctions with a gap longer than STRONG_GAP_S are unverifiable by motion
# alone (the camera panned away), so they additionally need *positive*
# appearance evidence (>= SMIN_STRONG) instead of just "not obviously
# different". SLOCAL_MIN is a last flagrance check on the crops nearest to
# the junction itself (last/first ~2 s of each fragment): it catches
# fragments whose overall colour matches but that visibly follow a
# different player right at the joint (ByteTrack can switch players
# mid-track, so a fragment's average appearance can lie about its ends).
DEFAULT_SMIN = 0.30
SMIN_STRONG = 0.45
STRONG_GAP_S = 5.0
SLOCAL_MIN = 0.15
LOCAL_WINDOW_S = 2.0
N_LOCAL_CROPS = 4

# Gate 3 details. Velocity is estimated over the fragment's last VEL_WINDOW
# frames and capped at a sprint (no player outruns MAX_SPEED); it is only
# extrapolated for up to VEL_PROJECT_MAX_S seconds into the gap -- beyond
# that a linear projection is fiction. Longer gaps instead widen the allowed
# radius by DMAX_SLACK_PER_S per second (an untracked player wanders), capped
# at DMAX_CAP_CM so a 10 s gap can never justify a cross-pitch merge.
VEL_WINDOW = 12
MAX_SPEED_CM_S = 900.0        # ~9 m/s
VEL_PROJECT_MAX_S = 1.5
DMAX_SLACK_PER_S = 600.0      # ~6 m/s of unaccounted drift
DMAX_CAP_CM = 3500.0

# Appearance descriptor: mean HSV histogram over the N_CROPS *sharpest* of
# N_CANDIDATES torso crops sampled evenly along the fragment. Sharpness
# filtering matters: fragments die exactly when the camera pans, so crops
# near fragment edges are often motion-blurred colour smears that would
# poison the histogram. The torso window (central band of the bbox) keeps
# mostly shirt pixels and drops grass/legs/feet.
N_CROPS = 10
N_CANDIDATES = 24
HIST_BINS = (16, 8, 4)        # H, S, V
TORSO_X = (0.20, 0.80)        # fraction of bbox width kept
TORSO_Y = (0.10, 0.55)        # fraction of bbox height kept (head->waist)
MIN_CROP_PX = 6               # skip degenerate boxes

CSV_FIELDS = ["Frame", "Object", "Object ID", "Team",
              "X1", "Y1", "X2", "Y2", "X_Pitch", "Y_Pitch"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

class Track:
    """One raw ByteTrack fragment: every row of a single ``Object ID``."""

    __slots__ = ("tid", "frames", "pitch", "bboxes", "team_votes",
                 "start_frame", "end_frame", "start_pos", "end_pos",
                 "vel_out", "team", "descriptor", "head_local", "tail_local")

    def __init__(self, tid):
        self.tid = tid
        self.frames = []        # sorted frame numbers
        self.pitch = []         # (x, y) cm, aligned with frames
        self.bboxes = []        # (x1, y1, x2, y2) image px, aligned
        self.team_votes = Counter()
        self.descriptor = None  # whole-fragment appearance
        self.head_local = None  # appearance of the first ~2 s
        self.tail_local = None  # appearance of the last ~2 s

    def finalize(self):
        order = np.argsort(self.frames)
        self.frames = [self.frames[i] for i in order]
        self.pitch = [self.pitch[i] for i in order]
        self.bboxes = [self.bboxes[i] for i in order]
        self.start_frame = self.frames[0]
        self.end_frame = self.frames[-1]
        self.start_pos = np.asarray(self.pitch[0], dtype=float)
        self.end_pos = np.asarray(self.pitch[-1], dtype=float)
        self.vel_out = _exit_velocity(self.frames, self.pitch)
        self.team = self.team_votes.most_common(1)[0][0] if self.team_votes else ""


def _exit_velocity(frames, pitch):
    """cm/frame velocity over the last ~VEL_WINDOW frames of a fragment."""
    if len(frames) < 2:
        return np.zeros(2)
    last = frames[-1]
    i = len(frames) - 1
    while i > 0 and last - frames[i - 1] <= VEL_WINDOW:
        i -= 1
    df = frames[-1] - frames[i]
    if df <= 0:
        return np.zeros(2)
    return (np.asarray(pitch[-1], float) - np.asarray(pitch[i], float)) / df


def load_tracks(csv_path):
    """Parse the tracking CSV into per-ID ``Track`` fragments.

    Returns ``(tracks, rows)`` where ``tracks`` maps Object ID -> Track and
    ``rows`` is every CSV row (players and ball) for later rewriting.
    """
    tracks = {}
    rows = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
            if row["Object"] != "player" or not row["Object ID"]:
                continue
            tid = row["Object ID"]
            tr = tracks.get(tid)
            if tr is None:
                tr = tracks[tid] = Track(tid)
            tr.frames.append(int(row["Frame"]))
            tr.pitch.append((float(row["X_Pitch"]), float(row["Y_Pitch"])))
            tr.bboxes.append((float(row["X1"]), float(row["Y1"]),
                              float(row["X2"]), float(row["Y2"])))
            if row["Team"] != "":
                tr.team_votes[row["Team"]] += 1
    for tr in tracks.values():
        tr.finalize()
    return tracks, rows


# --------------------------------------------------------------------------
# Appearance descriptors
# --------------------------------------------------------------------------

def _torso_crop(frame_img, bbox):
    """Crop the shirt region of a player bbox; None if degenerate."""
    h, w = frame_img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(round(x1 + TORSO_X[0] * bw))
    cx2 = int(round(x1 + TORSO_X[1] * bw))
    cy1 = int(round(y1 + TORSO_Y[0] * bh))
    cy2 = int(round(y1 + TORSO_Y[1] * bh))
    cx1, cx2 = max(0, cx1), min(w, cx2)
    cy1, cy2 = max(0, cy1), min(h, cy2)
    if cx2 - cx1 < MIN_CROP_PX or cy2 - cy1 < MIN_CROP_PX:
        return None
    return frame_img[cy1:cy2, cx1:cx2]


def _crop_histogram(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS),
                        [0, 180, 0, 256, 0, 256]).flatten()
    total = hist.sum()
    return hist / total if total > 0 else None


def hist_similarity(a, b):
    """Histogram intersection of two L1-normalised histograms, in [0, 1]."""
    if a is None or b is None:
        return -1.0
    return float(np.minimum(a, b).sum())


def _crop_sharpness(crop):
    """Variance of the Laplacian: low = motion blur (pan), high = crisp."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def track_descriptor(video_path, track, n_crops=N_CROPS):
    """Mean HSV histogram of the sharpest torso crops of one track.

    Convenience wrapper (random seeks). For all tracks at once prefer
    ``compute_descriptors`` which decodes the video sequentially.
    """
    idx = np.linspace(0, len(track.frames) - 1,
                      min(N_CANDIDATES, len(track.frames)))
    wanted = {track.frames[int(i)]: track.bboxes[int(i)] for i in idx}
    cap = cv2.VideoCapture(video_path)
    scored = []
    try:
        for frame_no, bbox in sorted(wanted.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
            ok, img = cap.read()
            if not ok:
                continue
            crop = _torso_crop(img, bbox)
            if crop is None:
                continue
            h = _crop_histogram(crop)
            if h is not None:
                scored.append((_crop_sharpness(crop), h))
    finally:
        cap.release()
    return _best_mean_hist(scored, n_crops)


def _best_mean_hist(scored, n_crops=N_CROPS):
    """Mean of the ``n_crops`` sharpest (sharpness, hist) pairs."""
    if not scored:
        return None
    scored = sorted(scored, key=lambda s: -s[0])[:n_crops]
    m = np.mean([h for _s, h in scored], axis=0)
    total = m.sum()
    return m / total if total > 0 else None


def read_tracking_meta(tracking_csv):
    """(fps, frame_stride) desde el .meta.json que escribe main.py.

    Sin esto el ReID hereda dos bugs con ``--frame-stride``:
      * ``--tmax`` se convierte a frames con un fps equivocado, y el hueco
        maximo permitido se distorsiona (a 15 fps reales leidos como 24, 30 s
        se vuelven 48 s) -> sobre-fusion, que es el peor error posible aca.
      * el frame N del CSV NO es el frame N del video, asi que los recortes de
        apariencia saldrian de fotogramas equivocados.
    """
    meta_path = tracking_csv.rsplit(".", 1)[0] + ".meta.json"
    if not os.path.exists(meta_path):
        print(f"AVISO: no encuentro {meta_path}; asumo {DEFAULT_FPS:.0f} fps y "
              f"stride 1. Si el tracking uso --frame-stride, esto esta mal.")
        return DEFAULT_FPS, 1
    with open(meta_path) as fh:
        meta = json.load(fh)
    fps = float(meta.get("effective_fps") or DEFAULT_FPS)
    stride = int(meta.get("frame_stride") or 1)
    print(f"Sidecar: effective_fps={fps:g}, frame_stride={stride}")
    return fps, stride


def compute_descriptors(video_path, tracks, n_crops=N_CROPS, fps=DEFAULT_FPS,
                        frame_stride=1):
    """Descriptor for every track in ONE sequential pass over the video.

    Random seeking per track is painfully slow on CPU; instead we mark which
    (frame -> [(tid, bbox)]) pairs we need, then decode the video once.
    Samples ``N_CANDIDATES`` crops per track (plus extra candidates inside
    the first/last ``LOCAL_WINDOW_S`` seconds) and keeps the sharpest
    (motion-blurred crops carry no shirt colour). Assigns
    ``track.descriptor`` plus the junction-local ``track.head_local`` /
    ``track.tail_local`` in place.
    """
    local_frames = int(LOCAL_WINDOW_S * fps)
    needed = defaultdict(list)   # frame number (1-based) -> [(tid, bbox)]
    for tr in tracks.values():
        n = len(tr.frames)
        want = {int(round(i))
                for i in np.linspace(0, n - 1, min(N_CANDIDATES, n))}
        # Dense candidates near both ends for the junction-local descriptors.
        head_end = 0
        while head_end < n - 1 and tr.frames[head_end + 1] - tr.frames[0] <= local_frames:
            head_end += 1
        tail_start = n - 1
        while tail_start > 0 and tr.frames[-1] - tr.frames[tail_start - 1] <= local_frames:
            tail_start -= 1
        want |= {int(round(i)) for i in np.linspace(0, head_end, min(8, head_end + 1))}
        want |= {int(round(i)) for i in np.linspace(tail_start, n - 1, min(8, n - tail_start))}
        for i in want:
            needed[tr.frames[i]].append((tr.tid, tr.bboxes[i]))

    scored = defaultdict(list)   # tid -> [(sharpness, hist, frame)]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        last_needed = max(needed)
        # El CSV numera los frames PROCESADOS (1,2,3...). Con stride S, el
        # frame k del CSV es el frame (k-1)*S+1 del video, asi que el video se
        # recorre entero y solo se usan los fotogramas que le corresponden.
        last_video_frame = (last_needed - 1) * frame_stride + 1
        video_frame = 1
        while video_frame <= last_video_frame:
            ok, img = cap.read()
            if not ok:
                break
            if (video_frame - 1) % frame_stride == 0:
                csv_frame = (video_frame - 1) // frame_stride + 1
                for tid, bbox in needed.get(csv_frame, ()):
                    crop = _torso_crop(img, bbox)
                    if crop is None:
                        continue
                    h = _crop_histogram(crop)
                    if h is not None:
                        scored[tid].append((_crop_sharpness(crop), h, csv_frame))
            video_frame += 1
    finally:
        cap.release()

    missing = 0
    for tr in tracks.values():
        sc = scored.get(tr.tid, [])
        tr.descriptor = _best_mean_hist([s[:2] for s in sc], n_crops)
        tr.head_local = _best_mean_hist(
            [s[:2] for s in sc if s[2] - tr.start_frame <= local_frames],
            N_LOCAL_CROPS)
        tr.tail_local = _best_mean_hist(
            [s[:2] for s in sc if tr.end_frame - s[2] <= local_frames],
            N_LOCAL_CROPS)
        missing += tr.descriptor is None
    if missing:
        print(f"  ⚠️  {missing} track(s) without a usable appearance crop "
              f"(they will never merge: appearance gate fails closed).")


# --------------------------------------------------------------------------
# Merging (union-find over the five gates)
# --------------------------------------------------------------------------

class _Group:
    """Current state of a merged set of fragments (a union-find root)."""

    __slots__ = ("members", "intervals", "start_frame", "end_frame",
                 "start_pos", "end_pos", "vel_out", "team_votes",
                 "descriptor", "head_desc", "tail_desc",
                 "head_local", "tail_local", "n_frames")

    def __init__(self, tr):
        self.members = [tr.tid]
        self.intervals = [(tr.start_frame, tr.end_frame)]
        self.start_frame = tr.start_frame
        self.end_frame = tr.end_frame
        self.start_pos = tr.start_pos
        self.end_pos = tr.end_pos
        self.vel_out = tr.vel_out
        self.team_votes = Counter(tr.team_votes)
        self.descriptor = tr.descriptor
        # Descriptors of the first/last member fragment: the appearance gate
        # compares the two fragments that actually touch at a junction (the
        # group average would dilute a bad joint into acceptability).
        self.head_desc = tr.descriptor
        self.tail_desc = tr.descriptor
        self.head_local = tr.head_local
        self.tail_local = tr.tail_local
        self.n_frames = len(tr.frames)

    @property
    def team(self):
        return self.team_votes.most_common(1)[0][0] if self.team_votes else ""

    def absorb(self, other):
        """Merge ``other`` (which starts after this group ends) into this."""
        assert self.end_frame < other.start_frame
        self.members += other.members
        self.intervals += other.intervals
        self.end_frame = other.end_frame
        self.end_pos = other.end_pos
        self.vel_out = other.vel_out
        self.tail_desc = other.tail_desc
        self.tail_local = other.tail_local
        self.team_votes += other.team_votes
        if self.descriptor is None or other.descriptor is None:
            self.descriptor = None
        else:
            w1, w2 = self.n_frames, other.n_frames
            d = (self.descriptor * w1 + other.descriptor * w2) / (w1 + w2)
            self.descriptor = d / d.sum()
        self.n_frames += other.n_frames


def _gate_check(a, b, tmax_frames, dmax, smin, fps):
    """All five gates between ordered groups a (earlier) and b (later).

    Returns ``(ok, cost)`` -- cost is only meaningful when ok, lower = safer.
    """
    # 1. temporally disjoint + 2. short gap
    gap = b.start_frame - a.end_frame
    if gap <= 0 or gap > tmax_frames:
        return False, None
    gap_s = gap / fps

    # 3. spatial plausibility: project the exit velocity (capped at sprint
    # speed, for at most VEL_PROJECT_MAX_S) and widen the radius with the gap.
    vel = a.vel_out.copy()
    speed = np.linalg.norm(vel) * fps
    if speed > MAX_SPEED_CM_S:
        vel *= MAX_SPEED_CM_S / speed
    horizon = min(gap, VEL_PROJECT_MAX_S * fps)
    predicted = a.end_pos + vel * horizon
    predicted[0] = np.clip(predicted[0], 0.0, PITCH_LENGTH_CM)
    predicted[1] = np.clip(predicted[1], 0.0, PITCH_WIDTH_CM)
    dist = float(np.linalg.norm(predicted - b.start_pos))
    allowed = min(dmax + DMAX_SLACK_PER_S * gap_s, DMAX_CAP_CM)
    if dist > allowed:
        return False, None

    # 4. same team (majority vote per group)
    if a.team != b.team:
        return False, None

    # 5. appearance veto, on the two fragments that touch at the junction
    # (fails closed when a descriptor is missing). Short gaps only need to
    # not look like a different shirt; long gaps need positive evidence;
    # and the crops nearest the joint itself must not be flagrantly
    # different (a fragment can drift onto another player near its end, so
    # its whole-fragment colour can lie about the junction).
    sim = hist_similarity(a.tail_desc, b.head_desc)
    if sim < smin:
        return False, None
    if gap_s > STRONG_GAP_S and sim < SMIN_STRONG:
        return False, None
    if hist_similarity(a.tail_local, b.head_local) < SLOCAL_MIN:
        return False, None

    cost = gap / tmax_frames + dist / allowed + (1.0 - sim)
    return True, cost


def merge_tracks(tracks, tmax_s=DEFAULT_TMAX_S, dmax=DEFAULT_DMAX_CM,
                 smin=DEFAULT_SMIN, fps=DEFAULT_FPS, verbose=True):
    """Union-find fixpoint over the five gates.

    Returns ``mapping``: original Object ID -> (merged Object ID, team),
    where the merged ID is the smallest original ID in the group and the team
    is the group's majority vote.
    """
    tmax_frames = int(round(tmax_s * fps))
    parent = {tid: tid for tid in tracks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    groups = {tid: _Group(tr) for tid, tr in tracks.items()}

    n_pass = 0
    while True:
        n_pass += 1
        roots = sorted({find(t) for t in tracks},
                       key=lambda r: groups[r].start_frame)
        # Collect every gate-passing ordered pair among current groups.
        candidates = []
        for i, ra in enumerate(roots):
            a = groups[ra]
            for rb in roots[i + 1:]:
                b = groups[rb]
                if b.start_frame - a.end_frame > tmax_frames:
                    break  # roots sorted by start; later ones only further
                ok, cost = _gate_check(a, b, tmax_frames, dmax, smin, fps)
                if ok:
                    candidates.append((cost, ra, rb))
        candidates.sort(key=lambda c: c[0])

        merged_this_pass = 0
        for _cost, ra, rb in candidates:
            ra, rb = find(ra), find(rb)
            if ra == rb:
                continue
            a, b = groups[ra], groups[rb]
            if a.start_frame > b.start_frame:
                a, b, ra, rb = b, a, rb, ra
            # Re-validate against the *current* group state: earlier merges
            # in this pass may have extended either group.
            ok, _ = _gate_check(a, b, tmax_frames, dmax, smin, fps)
            if not ok:
                continue
            a.absorb(b)
            parent[rb] = ra
            del groups[rb]
            merged_this_pass += 1
        if verbose:
            print(f"  pass {n_pass}: {merged_this_pass} merge(s), "
                  f"{len(groups)} identities")
        if merged_this_pass == 0:
            break

    mapping = {}
    for root, g in groups.items():
        merged_id = str(min(int(m) for m in g.members))
        for m in g.members:
            mapping[m] = (merged_id, g.team)
    return mapping


# --------------------------------------------------------------------------
# Output + validation
# --------------------------------------------------------------------------

def write_merged_csv(rows, mapping, output_path):
    """Rewrite player ``Object ID``/``Team`` with merged identity; ball as-is."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            if row["Object"] == "player" and row["Object ID"] in mapping:
                merged_id, team = mapping[row["Object ID"]]
                row = dict(row, **{"Object ID": merged_id, "Team": team})
            writer.writerow(row)
    return output_path


def check_overlaps(rows, mapping):
    """Frames where two source fragments of one merged ID coexist.

    Must be 0: an overlap means two players visible at once were fused.
    Returns ``(n_overlapping_frame_pairs, offending merged ids)``.
    """
    seen = {}
    overlaps = 0
    bad = set()
    for row in rows:
        if row["Object"] != "player" or row["Object ID"] not in mapping:
            continue
        key = (mapping[row["Object ID"]][0], row["Frame"])
        if key in seen and seen[key] != row["Object ID"]:
            overlaps += 1
            bad.add(key[0])
        else:
            seen[key] = row["Object ID"]
    return overlaps, sorted(bad, key=int)


def summarize(tracks, mapping):
    groups = defaultdict(list)
    for tid, (mid, _team) in mapping.items():
        groups[mid].append(tid)
    n_before, n_after = len(tracks), len(groups)
    print(f"\nIDs de jugador: {n_before} -> {n_after} "
          f"({n_before - n_after} fusiones)")
    merged = sorted(((mid, tids) for mid, tids in groups.items()
                     if len(tids) > 1), key=lambda g: -len(g[1]))
    if merged:
        print("Grupos fusionados más grandes:")
        for mid, tids in merged[:10]:
            frames = sum(len(tracks[t].frames) for t in tids)
            print(f"  ID {mid}: {len(tids)} fragmentos "
                  f"({', '.join(sorted(tids, key=int))}) — {frames} frames")
    return n_before, n_after


def render_merged_tracks(video_path, tracks, mapping, output_path,
                         merged_ids=None, top=6, max_frames=None):
    """Paint merged tracks on the video for visual verification.

    Draws the bbox of every fragment belonging to the selected merged IDs,
    one colour per merged ID, labelled ``M<merged> (<original>)`` -- so a
    correct merge shows the SAME colour following the SAME player across
    fragment boundaries. Defaults to the ``top`` merged IDs with the most
    fragments.
    """
    groups = defaultdict(list)
    for tid, (mid, _team) in mapping.items():
        groups[mid].append(tid)
    if merged_ids is None:
        merged_ids = [mid for mid, tids in
                      sorted(groups.items(), key=lambda g: -len(g[1]))
                      if len(tids) > 1][:top]
    merged_ids = [str(m) for m in merged_ids]

    palette = [(0, 215, 255), (255, 144, 30), (60, 20, 220), (50, 205, 50),
               (255, 0, 255), (0, 69, 255), (255, 255, 0), (147, 20, 255),
               (0, 128, 255), (255, 105, 180)]
    colors = {mid: palette[i % len(palette)] for i, mid in enumerate(merged_ids)}

    # frame -> [(bbox, merged id, original id)]
    per_frame = defaultdict(list)
    for mid in merged_ids:
        for tid in groups.get(mid, ()):
            tr = tracks[tid]
            for f, bbox in zip(tr.frames, tr.bboxes):
                per_frame[f].append((bbox, mid, tid))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    try:
        frame_no = 1
        last = max(per_frame) if per_frame else 0
        if max_frames:
            last = min(last, max_frames)
        while frame_no <= last:
            ok, img = cap.read()
            if not ok:
                break
            for bbox, mid, tid in per_frame.get(frame_no, ()):
                x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                c = colors[mid]
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
                cv2.putText(img, f"M{mid} ({tid})", (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            writer.write(img)
            frame_no += 1
    finally:
        cap.release()
        writer.release()
    print(f"Video de verificación: {output_path} "
          f"(IDs pintados: {', '.join(merged_ids)})")
    return output_path


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_reid(tracking_csv, video, output, tmax=DEFAULT_TMAX_S,
             dmax=DEFAULT_DMAX_CM, smin=DEFAULT_SMIN, fps=None,
             render=None, render_ids=None, render_top=6, verbose=True):
    """Full ReID pipeline: load -> descriptors -> merge -> write -> validate.

    Returns ``(n_ids_before, n_ids_after, n_overlaps)``.
    """
    print(f"ReID: {tracking_csv}")
    # El fps y el stride mandan: definen cuantos frames son ``tmax`` segundos y
    # que fotograma del video corresponde a cada frame del CSV. Un --fps
    # explicito pisa el sidecar (escape hatch), pero por defecto se lee.
    meta_fps, frame_stride = read_tracking_meta(tracking_csv)
    if fps is None:
        fps = meta_fps
    tracks, rows = load_tracks(tracking_csv)
    print(f"  {len(tracks)} tracks de jugador, "
          f"{sum(len(t.frames) for t in tracks.values())} filas")

    print("Extrayendo descriptores de apariencia (una pasada por el video)...")
    compute_descriptors(video, tracks, fps=fps, frame_stride=frame_stride)

    print(f"Fusionando (tmax={tmax}s, dmax={dmax}cm, smin={smin})...")
    mapping = merge_tracks(tracks, tmax_s=tmax, dmax=dmax, smin=smin,
                           fps=fps, verbose=verbose)

    write_merged_csv(rows, mapping, output)
    print(f"CSV fusionado: {output}")

    n_before, n_after = summarize(tracks, mapping)
    overlaps, bad = check_overlaps(rows, mapping)
    if overlaps:
        print(f"❌ {overlaps} solapamiento(s) temporal(es) en IDs: {bad} — "
              f"hay merges incorrectos, bajá tmax/dmax o subí smin.")
    else:
        print("✅ 0 solapamientos temporales (ningún merge junta dos "
              "jugadores visibles a la vez).")

    if render:
        render_merged_tracks(video, tracks, mapping, render,
                             merged_ids=render_ids, top=render_top)
    return n_before, n_after, overlaps


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge fragmented player IDs in a tracking CSV (ReID "
                    "post-process; spatio-temporal + team + HSV appearance).")
    p.add_argument("--tracking-csv", required=True,
                   help="Raw tracking CSV from data_cleanup/main.py.")
    p.add_argument("--video", required=True,
                   help="Source video (for appearance crops).")
    p.add_argument("--output", required=True,
                   help="Path for the merged CSV.")
    p.add_argument("--tmax", type=float, default=DEFAULT_TMAX_S,
                   help=f"Max gap between fragments, seconds "
                        f"(default {DEFAULT_TMAX_S}).")
    p.add_argument("--dmax", type=float, default=DEFAULT_DMAX_CM,
                   help=f"Base spatial tolerance in cm at gap~0; grows "
                        f"{DMAX_SLACK_PER_S:.0f} cm/s with the gap, capped at "
                        f"{DMAX_CAP_CM:.0f} (default {DEFAULT_DMAX_CM}).")
    p.add_argument("--smin", type=float, default=DEFAULT_SMIN,
                   help=f"Min appearance similarity 0..1 "
                        f"(default {DEFAULT_SMIN}).")
    p.add_argument("--fps", type=float, default=None,
                   help=f"Frames per second, converts --tmax to frames "
                        f"(default {DEFAULT_FPS:.0f}).")
    p.add_argument("--render", default=None, metavar="OUT.mp4",
                   help="Also write a video with sample merged tracks "
                        "painted for visual verification.")
    p.add_argument("--render-ids", default=None,
                   help="Comma-separated merged IDs to paint "
                        "(default: the biggest merged groups).")
    p.add_argument("--render-top", type=int, default=6,
                   help="How many merged groups to paint (default 6).")
    return p.parse_args()


def main():
    args = parse_args()
    render_ids = args.render_ids.split(",") if args.render_ids else None
    run_reid(args.tracking_csv, args.video, args.output,
             tmax=args.tmax, dmax=args.dmax, smin=args.smin, fps=args.fps,
             render=args.render, render_ids=render_ids,
             render_top=args.render_top)


if __name__ == "__main__":
    main()
