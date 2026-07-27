"""
Command-line tracking entry point.

This is a script version of ``track/track.ipynb``, using **ultralytics YOLOv8**
for detection (no Roboflow API key required). It runs YOLOv8 over a video to
detect players and the ball, and writes raw tracking data (player & ball pitch
coordinates per frame) to a CSV that the rest of the pipeline (``data_cleanup``
cleaning and ``event_generation``) can consume.

Example:
    python data_cleanup/main.py \\
        --video ./track/footage/2e57b9_0.mp4 \\
        --output ./track/output

The CSV is written to ``<output>/<video name>.csv`` in the "raw" format read by
``Match.import_raw_data``.

Note: a generic COCO ``yolov8n.pt`` detects people ("player") and the
"sports ball" but has no pitch keypoint model, so pitch coordinates are taken
directly from image space (each detection's bottom-centre pixel, normalised to
the standard pitch dimensions) rather than via a homography. The coordinates are
therefore in camera perspective and approximate. Point ``--player-model`` /
``--ball-model`` at a football-trained ``.pt`` for better detections.

Ball detection notes:
    * The ball is a tiny, fast object that the nano COCO model misses
      constantly. By default ``--ball-model football`` downloads a
      football-specific YOLOv8 checkpoint trained on broadcast footage (classes
      ``ball/goalkeeper/player/referee``) from the Hugging Face Hub -- no API key
      required -- and falls back to ``yolov8m.pt`` (medium, far better small
      object recall than nano) if that download is unavailable.
    * Ball confidence defaults to a low ``0.15`` so faint/blurred balls are not
      discarded.
    * Short detection gaps are linearly interpolated (see ``interpolate_ball``)
      so a few missed frames don't reset possession downstream.
"""

import argparse
import json
import os

import numpy as np
from tqdm import tqdm


def _patch_torch_load_for_ultralytics():
    """Allow ultralytics ``.pt`` checkpoints to load under PyTorch >= 2.6.

    PyTorch 2.6 flipped ``torch.load``'s default to ``weights_only=True``, which
    refuses to unpickle the ultralytics model globals (``DetectionModel`` etc.)
    and breaks ``YOLO(weights)`` on older ultralytics releases. Our checkpoints
    come from trusted sources (official ultralytics weights and a public
    football model on the HF Hub), so we restore the full-unpickle behaviour.
    """
    try:
        import torch
    except Exception:
        return
    if getattr(torch.load, "_ultralytics_patched", False):
        return
    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _load._ultralytics_patched = True
    torch.load = _load


_patch_torch_load_for_ultralytics()

# Models. The sentinel ``"football"`` resolves (in ``resolve_model_path``) to a
# football-trained YOLOv8 checkpoint downloaded from the Hugging Face Hub
# (classes: ball/goalkeeper/player/referee), with a fallback to a stock
# ultralytics model. Pass any ``.pt`` path to override.
#
# Using the football model for BOTH players and the ball gives far more
# consistent player detection on broadcast footage than the generic COCO
# ``yolov8n.pt`` -- which fragments tracks badly (thousands of short-lived IDs)
# and can't separate players from referees. When both resolve to the same
# weights we run a single inference per frame instead of two.
DEFAULT_PLAYER_MODEL = "football"
FALLBACK_PLAYER_MODEL = "yolov8n.pt"   # generic COCO "person" detector
DEFAULT_BALL_MODEL = "football"
FALLBACK_BALL_MODEL = "yolov8m.pt"     # medium COCO model: best small-object recall

# Public, football-trained YOLOv8 weights (Roboflow "football-players-detection"
# dataset, broadcast footage; classes: ball/goalkeeper/player/referee).
# ``best.pt`` is a standard ultralytics checkpoint, downloadable without an API
# key. mAP@0.5 ~0.785. Loads directly via ``YOLO(path)``.
FOOTBALL_MODEL_REPO = "uisikdag/yolo-v8-football-players-detection"
FOOTBALL_MODEL_FILE = "best.pt"

# The ball is small and easily missed; keep its confidence low.
DEFAULT_BALL_CONF = 0.10
DEFAULT_PLAYER_CONF = 0.3

# Inference resolution. The ball is tiny, so detecting it at higher resolution
# (e.g. 1280 instead of the 640 default) recovers many small-object detections
# the model otherwise misses. Higher = better ball recall but slower.
DEFAULT_IMGSZ = 1280

# Pitch keypoint model (YOLOv8-pose, 32 keypoints matching SoccerPitchConfiguration).
# Enables a real image->pitch homography so event coordinates are true pitch
# positions instead of a naive camera-perspective scaling. ``"football-field"``
# downloads the weights from the HF Hub (no API key); ``"none"`` disables
# homography and falls back to image-space coordinates.
DEFAULT_PITCH_MODEL = "football-field"
PITCH_MODEL_REPO = "martinjolif/yolo-football-pitch-detection"
PITCH_MODEL_FILE = "yolo-football-pitch-detection.pt"
# Minimum keypoint confidence to use a landmark in the homography, and minimum
# number of confident landmarks needed to solve one.
DEFAULT_PITCH_CONF = 0.5
MIN_HOMOGRAPHY_POINTS = 4
# Recompute the homography every N frames (the camera pans smoothly, so we reuse
# the last transform in between to avoid a keypoint pass on every frame).
DEFAULT_HOMOGRAPHY_EVERY = 5

# Pitch keypoints are large features and gain nothing from very high resolution,
# so the pitch model is capped here even when players/ball run at a higher imgsz
# (keeps precision where it matters without paying the cost where it doesn't).
DEFAULT_PITCH_IMGSZ = 1280

# Linearly interpolate ball position across detection gaps no longer than this
# many frames (longer gaps stay empty). Prevents possession from resetting on
# every missed detection.
# Raised from 15 once the continuity gate started rejecting bad detections:
# the gate trades coverage for trustworthiness (ball presence 87% -> 74%), and
# interpolation is the safe way to buy that coverage back because a linear fill
# is SMOOTH and cannot reintroduce teleports. Measured on spain-france at
# stride 2 (so the effective gap is half this): ball presence 74.4% -> 90.1%,
# passes 478 -> 594, set pieces 131 -> 125 (slightly better, not worse).
BALL_INTERP_MAX_GAP = 50

# Ball-candidate continuity (see ``_pick_ball``). A powerful shot peaks around
# 35 m/s; above that no real ball is moving, so a candidate that far from the
# last known position is a different white object (penalty spot, line, crowd).
MAX_BALL_SPEED_MS = 40.0
# The gate compares distances in PITCH METRES (see _pick_ball), so no pixel
# scale and no slack term are involved. Two earlier attempts failed here:
# a 120 px "jitter slack" was ~6.6 m of pitch and by itself licensed ~95 m/s
# jumps (impossible moves only fell 26.3% -> 17.6%), and gating in pixel space
# stalled at 12.1% because perspective makes a pixel worth far more metres at
# the far touchline. Sweep on psg-bayern in pitch space: 40 m/s keeps ~57% of
# detections at ~2% impossible; 30 m/s reaches 0% but costs 8 more points of
# coverage, which is not worth it (interpolate_ball bridges short gaps anyway).
DEFAULT_TRACK_FPS = 15.0    # only a fallback; the real effective fps is passed in

# ByteTrack keeps a "lost" player track alive for this many frames before
# dropping it; a longer buffer reuses the same id across occlusions/missed
# detections instead of minting a fresh id on every reappearance (which
# fragmented players into thousands of short-lived ids).
DEFAULT_LOST_TRACK_BUFFER = 150

# Player tracks shorter than this many frames are treated as detector/tracker
# noise and discarded before event generation.
DEFAULT_MIN_TRACK_FRAMES = 12

# COCO class ids produced by the stock yolov8 weights.
COCO_PERSON = 0
COCO_SPORTS_BALL = 32

# Standard pitch dimensions (centimetres) used to scale normalised image-space
# coordinates. Matches the dimensions assumed by ``lib.pitch`` for "raw" data.
PITCH_LENGTH_CM = 12000.0
PITCH_WIDTH_CM = 7000.0

# On frames with few/poor pitch keypoints the homography extrapolates wildly and
# sends detections far off the pitch. These margins (fractions of pitch size)
# bound that noise: the ball is dropped (and later interpolated) when beyond
# BALL, players are clamped into the PLAYER band.
BALL_PITCH_MARGIN = 0.10
PLAYER_PITCH_MARGIN = 0.15


def _pitch_in_bounds(xy, margin):
    return (-margin * PITCH_LENGTH_CM <= xy[0] <= (1 + margin) * PITCH_LENGTH_CM and
            -margin * PITCH_WIDTH_CM <= xy[1] <= (1 + margin) * PITCH_WIDTH_CM)


def _clamp_pitch(arr, margin):
    if len(arr) == 0:
        return arr
    arr = np.asarray(arr, dtype=float).copy()
    arr[:, 0] = np.clip(arr[:, 0], -margin * PITCH_LENGTH_CM, (1 + margin) * PITCH_LENGTH_CM)
    arr[:, 1] = np.clip(arr[:, 1], -margin * PITCH_WIDTH_CM, (1 + margin) * PITCH_WIDTH_CM)
    return arr


def image_to_pitch(detections, frame_w, frame_h):
    """Project detections to (approximate) pitch coordinates from image space.

    Uses each detection's bottom-centre pixel, normalised by the frame size and
    scaled to the standard pitch dimensions. No homography / keypoint model.
    """
    import supervision as sv

    if len(detections) == 0:
        return np.zeros((0, 2))
    anchors = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(float)
    pitch = np.empty_like(anchors)
    pitch[:, 0] = anchors[:, 0] / frame_w * PITCH_LENGTH_CM
    pitch[:, 1] = anchors[:, 1] / frame_h * PITCH_WIDTH_CM
    return pitch


def resolve_pitch_model_path(spec):
    """Resolve ``--pitch-model`` to a path, or None to disable homography."""
    if spec in (None, "none", "None", ""):
        return None
    if spec != "football-field":
        return spec
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=PITCH_MODEL_REPO, filename=PITCH_MODEL_FILE)
        print(f"Using pitch keypoint model {PITCH_MODEL_REPO} (homography enabled)")
        return path
    except Exception as exc:
        print(f"Could not fetch pitch model ({exc}); using image-space coords (no homography)")
        return None


def build_pitch_transformer(pitch_result, min_conf=DEFAULT_PITCH_CONF,
                            min_points=MIN_HOMOGRAPHY_POINTS):
    """Build an image->pitch ``ViewTransformer`` from detected pitch keypoints.

    Returns the transformer, or None when too few confident landmarks were
    found to solve a homography (caller then reuses the last good transform).
    """
    import supervision as sv
    try:
        from sports.configs.soccer import SoccerPitchConfiguration
        from sports.common.view import ViewTransformer
    except Exception:
        return None

    key_points = sv.KeyPoints.from_ultralytics(pitch_result)
    if key_points.xy is None or len(key_points.xy) == 0 or key_points.confidence is None:
        return None
    conf = key_points.confidence[0]
    mask = conf > min_conf
    if int(mask.sum()) < min_points:
        return None

    frame_pts = key_points.xy[0][mask].astype(np.float32)
    pitch_pts = np.array(SoccerPitchConfiguration().vertices)[mask].astype(np.float32)
    try:
        return ViewTransformer(source=frame_pts, target=pitch_pts)
    except Exception:
        return None


def anchors_to_pitch(detections, transformer):
    """Project detection anchors to pitch coords via a homography transform."""
    import supervision as sv

    if len(detections) == 0:
        return np.zeros((0, 2))
    xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(np.float32)
    return transformer.transform_points(points=xy).astype(float)


def resolve_model_path(spec, fallback):
    """Resolve a model spec (``--player-model`` / ``--ball-model``) to a path.

    ``"football"`` downloads the football-trained YOLOv8 checkpoint from the
    Hugging Face Hub (no API key) and returns its local path, falling back to
    ``fallback`` if the download is unavailable. Any other value is returned
    unchanged (an explicit ``.pt`` path or an ultralytics model name).
    """
    if spec != "football":
        return spec
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=FOOTBALL_MODEL_REPO, filename=FOOTBALL_MODEL_FILE)
        print(f"Using football-specific weights {FOOTBALL_MODEL_REPO}/{FOOTBALL_MODEL_FILE}")
        return path
    except Exception as exc:  # network down, hub error, or missing huggingface_hub
        print(f"Could not fetch football model ({exc}); falling back to {fallback}")
        return fallback


def resolve_ball_class_id(model):
    """Find the model's "ball" class id so detection works for any weights.

    Football models label the ball as class 0; stock COCO weights use 32. We
    read the class id from the model's ``names`` map instead of hard-coding it.
    """
    names = getattr(model, "names", None) or {}
    for class_id, name in names.items():
        if str(name).strip().lower() in ("ball", "sports ball", "soccer ball", "football"):
            return int(class_id)
    return COCO_SPORTS_BALL


def resolve_player_class_ids(model):
    """Class ids the tracker should treat as players (players + goalkeepers).

    Football models expose ball/goalkeeper/player/referee; we keep players and
    goalkeepers and drop the ball and the referee (so referees don't pollute the
    player tracks or the team classifier). Stock COCO weights only have "person"
    (id 0), which is then used as the fallback.
    """
    names = getattr(model, "names", None) or {}
    ids = [int(class_id) for class_id, name in names.items()
           if str(name).strip().lower() in ("player", "goalkeeper", "goal keeper", "person")]
    return ids or [COCO_PERSON]


def _pick_ball(ball_detections, ball_pitch, last_pitch, gap_frames,
               fps=DEFAULT_TRACK_FPS):
    """Choose one ball detection, preferring trajectory continuity.

    Works in PITCH COORDINATES (centimetres), never pixels. An earlier version
    gated in pixel space and only cut impossible moves 24.5% -> 12.1%, because
    the homography is perspective: the same pixel distance is a few metres near
    the camera and many metres at the far touchline, so a pixel radius is far
    too permissive exactly where play is furthest away. Distances are compared
    in metres, which is also the unit the "impossible move" metric uses.

    ``last_pitch`` is the previously accepted ball position (cm) or None, and
    ``gap_frames`` how many frames ago it was seen. Reachability is a HARD
    filter -- confidence only breaks ties among physically possible candidates,
    so a high-confidence pitch marking cannot outvote a faint real ball.
    Returns empty selections when nothing is reachable: leaving the frame blank
    (interpolate_ball bridges it) beats recording an impossible position.

    Always index Detections with a SLICE (``[i:i+1]``): supervision validates
    that ``xyxy`` stays 2-D (N, 4) and scalar indexing raises deep inside
    ``Detections.__post_init__``.
    """
    conf = ball_detections.confidence
    if len(ball_detections) == 0:
        return ball_detections, ball_pitch
    if conf is None:
        return ball_detections[0:1], ball_pitch[0:1]
    if last_pitch is None:
        best = int(np.argmax(conf))
        return ball_detections[best:best + 1], ball_pitch[best:best + 1]

    # Reachable radius in centimetres, growing with the blank gap so a
    # genuinely lost ball is re-acquired instead of suppressed forever.
    radius_cm = (MAX_BALL_SPEED_MS * max(1, gap_frames) / max(1.0, fps)) * 100.0

    reachable = []
    for i, (x, y) in enumerate(ball_pitch):
        dist = ((x - last_pitch[0]) ** 2 + (y - last_pitch[1]) ** 2) ** 0.5
        if dist <= radius_cm:
            reachable.append((float(conf[i]), -dist, i))

    if not reachable:
        return ball_detections[0:0], ball_pitch[0:0]

    best = max(reachable)[2]
    return ball_detections[best:best + 1], ball_pitch[best:best + 1]


def get_detections(frame, player_result, ball_result, tracker, team_classifier,
                   frame_w, frame_h, ball_class_id=COCO_SPORTS_BALL,
                   player_class_ids=(COCO_PERSON,), player_conf=DEFAULT_PLAYER_CONF,
                   transformer=None, last_ball_pitch=None, frames_since_ball=1,
                   effective_fps=DEFAULT_TRACK_FPS):
    import supervision as sv

    # Players: keep the player/goalkeeper classes (drop ball & referee). When a
    # single shared model is run at the low ball confidence, also drop player
    # boxes below ``player_conf`` so player behaviour stays unchanged.
    detections = sv.Detections.from_ultralytics(player_result)
    players_detections = detections[np.isin(detections.class_id, list(player_class_ids))]
    if players_detections.confidence is not None:
        players_detections = players_detections[players_detections.confidence >= player_conf]
    players_detections = players_detections.with_nms(threshold=0.5, class_agnostic=True)
    players_detections = tracker.update_with_detections(detections=players_detections)

    if team_classifier and len(players_detections):
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)
    else:
        players_detections.class_id = np.zeros(len(players_detections), dtype=int)

    # Ball: filter to the model's ball class (0 for football models, 32 for COCO).
    ball_all = sv.Detections.from_ultralytics(ball_result)
    ball_detections = ball_all[ball_all.class_id == ball_class_id]
    # The ball is unique, so only one box survives. Picking purely by
    # confidence is WRONG on a fixed camera: the penalty spot, pitch markings
    # and crowd objects are small white blobs that regularly out-score the real
    # ball, and each time they do the stored position teleports. Measured on
    # spain-france: 25% of consecutive ball movements were physically
    # impossible (p90 = 417 m/s = 1502 km/h), which is what makes possession
    # flicker and defeats any physics-based possession logic downstream.
    # So candidates are scored by CONTINUITY first: a candidate reachable from
    # the last known position at a plausible speed beats a more confident one
    # that would require teleporting.

    # Project to pitch coords. Image-space is always on-pitch but approximate;
    # the homography is accurate but goes wild on bad-keypoint frames. So: keep
    # the (approximate) image-space projection as a safety net and only use the
    # homography when it's trustworthy for this frame -- never dropping the ball.
    img_players = image_to_pitch(players_detections, frame_w, frame_h)
    img_ball = image_to_pitch(ball_detections, frame_w, frame_h)

    if transformer is None:
        players_pitch, ball_pitch = img_players, img_ball
    else:
        h_players = anchors_to_pitch(players_detections, transformer)
        h_ball = anchors_to_pitch(ball_detections, transformer)
        # If many players land off the pitch, the homography is unreliable this
        # frame -> fall back to image-space for everything (keeps it consistent).
        off_frac = (float(np.mean([not _pitch_in_bounds(p, PLAYER_PITCH_MARGIN)
                                   for p in h_players])) if len(h_players) else 0.0)
        if off_frac > 0.3:
            players_pitch, ball_pitch = img_players, img_ball
        else:
            players_pitch = _clamp_pitch(h_players, PLAYER_PITCH_MARGIN)
            # Keep the ball; if its homography point is wild, use image-space
            # (approximate) instead of discarding the detection.
            if len(h_ball) and not _pitch_in_bounds(h_ball[0], BALL_PITCH_MARGIN):
                ball_pitch = img_ball
            else:
                ball_pitch = h_ball

    # Continuity gate LAST, now that ball candidates have real pitch
    # coordinates: the physics of "how far can a ball travel" only makes sense
    # in metres, not pixels (see _pick_ball).
    ball_detections, ball_pitch = _pick_ball(
        ball_detections, np.asarray(ball_pitch, dtype=float).reshape(-1, 2),
        last_ball_pitch, frames_since_ball, fps=effective_fps)

    players_detections.data["pitch_xy"] = players_pitch
    ball_detections.data["pitch_xy"] = ball_pitch

    return players_detections, ball_detections


def generate_team_model(video_path, player_model, player_class_ids=(COCO_PERSON,),
                        stride=30, imgsz=DEFAULT_IMGSZ):
    import supervision as sv
    from sports.common.team import TeamClassifier

    frame_generator = sv.get_video_frames_generator(source_path=video_path, stride=stride)
    crops = []
    for frame in tqdm(frame_generator, desc="collecting crops"):
        result = player_model(frame, conf=DEFAULT_PLAYER_CONF, imgsz=imgsz, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[np.isin(detections.class_id, list(player_class_ids))]
        crops += [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

    team_classifier = TeamClassifier(device="cuda") if _cuda_available() else TeamClassifier()
    team_classifier.fit(crops)
    return team_classifier


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _ball_is_empty(entry):
    """A stored ball entry is "empty" when no detection was found that frame."""
    return (entry is None or
            (entry["X1"] == 0 and entry["Y1"] == 0 and
             entry["X2"] == 0 and entry["Y2"] == 0))


def interpolate_ball(ball, total_frames, max_gap=BALL_INTERP_MAX_GAP):
    """Linearly interpolate the ball across short detection gaps, in place.

    A "gap" is a run of consecutive frames with no ball detection bounded on
    both sides by a real detection. Gaps no longer than ``max_gap`` frames are
    filled by linearly interpolating every stored coordinate (bounding box +
    pitch position) between the last and next known frames; longer gaps, and
    gaps at the very start/end of the clip, are left empty. This keeps the ball
    "present" through brief misses so possession isn't reset downstream.

    Returns the number of frames filled.
    """
    keys = ["X1", "Y1", "X2", "Y2", "X_Pitch", "Y_Pitch"]
    filled = 0
    frame = 1
    while frame <= total_frames:
        start = ball.get(str(frame))
        if _ball_is_empty(start):
            frame += 1
            continue
        # Walk forward to the next detected frame.
        nxt = frame + 1
        while nxt <= total_frames and _ball_is_empty(ball.get(str(nxt))):
            nxt += 1
        gap = nxt - frame - 1  # number of missing frames between the two detections
        if nxt <= total_frames and 0 < gap <= max_gap:
            end = ball[str(nxt)]
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                ball[str(frame + k)] = {
                    key: start[key] + (end[key] - start[key]) * t for key in keys
                }
            filled += gap
        frame = nxt if nxt > frame else frame + 1
    return filled


def drop_short_tracks(players, min_frames):
    """Remove player tracks present in fewer than ``min_frames`` frames.

    Returns ``(kept_players, dropped_count)``. These ultra-short tracks are the
    fragments ByteTrack mints when it loses and re-IDs a player; dropping them
    cuts phantom turnovers in the events without touching real players.
    """
    if min_frames <= 1:
        return players, 0
    kept = {pid: data for pid, data in players.items() if len(data) >= min_frames}
    return kept, len(players) - len(kept)


def save_tracking_results(players, ball, frames, output_path):
    # Bucket player rows by frame in a single pass, then emit in frame order.
    # This is linear in the number of rows; the previous ``csv += ...`` in a
    # double loop was O(frames x players) with O(n^2) string growth, which
    # stalls on full 90-minute matches (~millions of rows).
    by_frame = {}
    for player_id, player_data in players.items():
        for f_str, d in player_data.items():
            by_frame.setdefault(int(f_str), []).append(",".join(map(str, [
                int(f_str), "player", player_id, d["Team"],
                d["X1"], d["Y1"], d["X2"], d["Y2"], d["X_Pitch"], d["Y_Pitch"],
            ])))

    rows = ["Frame,Object,Object ID,Team,X1,Y1,X2,Y2,X_Pitch,Y_Pitch"]
    for frame in range(1, frames):
        rows.extend(by_frame.get(frame, []))
        if str(frame) in ball:
            d = ball[str(frame)]
            rows.append(",".join(map(str, [
                frame, "ball", "", "",
                d["X1"], d["Y1"], d["X2"], d["Y2"], d["X_Pitch"], d["Y_Pitch"],
            ])))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(rows) + "\n")
    return output_path


def track(video_path, output_dir,
          player_model_path=DEFAULT_PLAYER_MODEL,
          ball_model_path=DEFAULT_BALL_MODEL,
          ball_conf=DEFAULT_BALL_CONF, ball_interp_gap=BALL_INTERP_MAX_GAP,
          track_buffer=DEFAULT_LOST_TRACK_BUFFER,
          min_track_frames=DEFAULT_MIN_TRACK_FRAMES, imgsz=DEFAULT_IMGSZ,
          pitch_model_path=DEFAULT_PITCH_MODEL, pitch_conf=DEFAULT_PITCH_CONF,
          homography_every=DEFAULT_HOMOGRAPHY_EVERY, pitch_imgsz=DEFAULT_PITCH_IMGSZ,
          track_teams=True, generate_video=False, stride=30, frame_stride=1):
    import supervision as sv
    from ultralytics import YOLO

    # Resolve and load the player model (football weights by default).
    resolved_player_path = resolve_model_path(player_model_path, FALLBACK_PLAYER_MODEL)
    try:
        player_model = YOLO(resolved_player_path)
    except Exception as exc:
        print(f"Failed to load player model '{resolved_player_path}' ({exc}); "
              f"using {FALLBACK_PLAYER_MODEL}")
        resolved_player_path = FALLBACK_PLAYER_MODEL
        player_model = YOLO(FALLBACK_PLAYER_MODEL)
    player_class_ids = resolve_player_class_ids(player_model)

    # Resolve the ball model. If it resolves to the same weights as the player
    # model, reuse it and run a single inference per frame.
    resolved_ball_path = resolve_model_path(ball_model_path, FALLBACK_BALL_MODEL)
    share_model = (resolved_ball_path == resolved_player_path)
    if share_model:
        ball_model = player_model
        print("Player and ball share one model: running a single inference per frame.")
    else:
        try:
            ball_model = YOLO(resolved_ball_path)
        except Exception as exc:
            print(f"Failed to load ball model '{resolved_ball_path}' ({exc}); "
                  f"using {FALLBACK_BALL_MODEL}")
            ball_model = YOLO(FALLBACK_BALL_MODEL)
    ball_class_id = resolve_ball_class_id(ball_model)

    # Pitch keypoint model for homography (optional; falls back to image space).
    pitch_model = None
    resolved_pitch_path = resolve_pitch_model_path(pitch_model_path)
    if resolved_pitch_path:
        try:
            pitch_model = YOLO(resolved_pitch_path)
        except Exception as exc:
            print(f"Failed to load pitch model '{resolved_pitch_path}' ({exc}); "
                  f"using image-space coords (no homography)")
            pitch_model = None

    ellipse_annotator = triangle_annotator = label_annotator = None
    if generate_video:
        ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']), thickness=2)
        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'), text_position=sv.Position.BOTTOM_CENTER)
        triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'), base=25, height=21, outline_thickness=1)

    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    frame_w, frame_h = video_info.width, video_info.height
    fps = int(round(video_info.fps)) or 24

    # ``frame_stride`` runs detection on every Nth video frame: N=2 halves the
    # GPU cost. The written frame numbers stay CONSECUTIVE (1..M over processed
    # frames) because lib/ball.py indexes ``frames[frame_number - 1]`` and
    # event_generator walks ``range(1, match.frames + 1)`` -- gaps would break
    # both. The real timeline is preserved via the sidecar ``effective_fps``.
    frame_stride = max(1, int(frame_stride))
    effective_fps = fps / frame_stride

    # Every frame-count parameter below is expressed in *processed* frames, so
    # dividing by the stride keeps its meaning in SECONDS unchanged.
    if frame_stride > 1:
        track_buffer = max(1, round(track_buffer / frame_stride))
        min_track_frames = max(1, round(min_track_frames / frame_stride))
        ball_interp_gap = max(1, round(ball_interp_gap / frame_stride))
        # ``homography_every`` is deliberately NOT rescaled: it stays in
        # processed frames, so the pitch model runs ``frame_stride`` times less
        # often in wall-clock terms. On a fixed/tactical camera the homography
        # barely moves, so that costs nothing and pays for the second detection
        # pass. Lower it manually if the camera pans a lot.
        print(f"frame-stride {frame_stride}: procesando 1 de cada {frame_stride} frames "
              f"({fps} fps -> {effective_fps:g} fps efectivos).")
        print(f"  reescalados (conservan su valor en segundos): "
              f"track-buffer={track_buffer}, min-track-frames={min_track_frames}, "
              f"ball-interp-gap={ball_interp_gap}")
        print(f"  homography-every={homography_every} sin reescalar "
              f"(refresca cada {homography_every * frame_stride} frames de video; "
              f"OK con camara fija)")

    # Longer lost-track buffer => fewer fragmented ids across occlusions.
    try:
        tracker = sv.ByteTrack(lost_track_buffer=track_buffer,
                               frame_rate=max(1, int(round(effective_fps))))
    except TypeError:  # older supervision signature without these kwargs
        tracker = sv.ByteTrack()
    tracker.reset()

    team_classifier = None
    if track_teams:
        try:
            team_classifier = generate_team_model(
                video_path, player_model, player_class_ids=player_class_ids,
                stride=stride, imgsz=imgsz)
        except Exception as exc:
            # The team classifier pulls a heavy umap/numba stack that can clash
            # with the runtime's NumPy. Degrade gracefully instead of crashing:
            # tracking still runs, players just aren't split into teams.
            print(f"⚠️  Team classifier unavailable ({exc}); continuing WITHOUT "
                  f"team labels (events will be less detailed).")
            team_classifier = None

    frame_generator = sv.get_video_frames_generator(video_path)

    players, ball = {}, {}
    frame_number = 1
    # Last accepted ball position (pitch cm) + how long ago, for _pick_ball.
    last_ball_pitch, frames_since_ball = None, 1
    transformer = None  # most recent image->pitch homography

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    annotated_video_path = os.path.join(output_dir, video_name + "_tracked.mp4")
    sink = sv.VideoSink(target_path=annotated_video_path, video_info=video_info) if generate_video else None
    if sink:
        sink.__enter__()

    shared_conf = min(DEFAULT_PLAYER_CONF, ball_conf)
    for source_index, frame in enumerate(tqdm(frame_generator, desc="Collecting Tracking Data...")):
        # Skip the frames the stride drops *before* any inference -- decoding is
        # cheap, the two YOLO passes are what costs.
        if source_index % frame_stride:
            continue

        if share_model:
            # One pass at the lower confidence; players are re-thresholded in
            # get_detections so their behaviour is unchanged.
            result = player_model(frame, conf=shared_conf, imgsz=imgsz, verbose=False)[0]
            player_result = ball_result = result
        else:
            player_result = player_model(frame, conf=DEFAULT_PLAYER_CONF, imgsz=imgsz, verbose=False)[0]
            ball_result = ball_model(frame, conf=ball_conf, imgsz=imgsz, verbose=False)[0]

        # Refresh the homography every ``homography_every`` frames; reuse the
        # last good transform in between (and keep it if a frame has too few
        # keypoints to solve a new one).
        if pitch_model is not None and (frame_number - 1) % homography_every == 0:
            pitch_result = pitch_model(frame, imgsz=min(pitch_imgsz, imgsz), verbose=False)[0]
            new_transformer = build_pitch_transformer(pitch_result, min_conf=pitch_conf)
            if new_transformer is not None:
                transformer = new_transformer

        all_detections, ball_detections = get_detections(
            frame, player_result, ball_result, tracker, team_classifier, frame_w, frame_h,
            ball_class_id=ball_class_id, player_class_ids=player_class_ids,
            player_conf=DEFAULT_PLAYER_CONF, transformer=transformer,
            last_ball_pitch=last_ball_pitch, frames_since_ball=frames_since_ball,
            effective_fps=effective_fps)

        object_ids = all_detections.tracker_id
        team_ids = all_detections.class_id
        pitch_xys = all_detections.data["pitch_xy"]
        ball_pitch_xys = ball_detections.data["pitch_xy"]
        all_detections.class_id = all_detections.class_id.astype(int)

        for idx, xyxy in enumerate(all_detections.xyxy):
            object_id = str(object_ids[idx])
            players.setdefault(object_id, {})
            players[object_id][str(frame_number)] = {
                "Team": team_ids[idx],
                "X1": xyxy[0], "Y1": xyxy[1], "X2": xyxy[2], "Y2": xyxy[3],
                "X_Pitch": pitch_xys[idx][0], "Y_Pitch": pitch_xys[idx][1],
            }

        if ball_detections.xyxy.shape[0]:
            b = ball_detections
            last_ball_pitch = (ball_pitch_xys[0][0], ball_pitch_xys[0][1])
            frames_since_ball = 1
            ball[str(frame_number)] = {
                "X1": b.xyxy[0][0], "Y1": b.xyxy[0][1], "X2": b.xyxy[0][2], "Y2": b.xyxy[0][3],
                "X_Pitch": ball_pitch_xys[0][0], "Y_Pitch": ball_pitch_xys[0][1],
            }
        else:
            frames_since_ball += 1
            ball[str(frame_number)] = {
                "X1": 0, "Y1": 0, "X2": 0, "Y2": 0, "X_Pitch": 0, "Y_Pitch": 0,
            }

        if generate_video:
            labels = [f"#{tid}" for tid in all_detections.tracker_id]
            annotated = frame.copy()
            annotated = ellipse_annotator.annotate(scene=annotated, detections=all_detections)
            annotated = label_annotator.annotate(scene=annotated, detections=all_detections, labels=labels)
            annotated = triangle_annotator.annotate(scene=annotated, detections=ball_detections)
            sink.write_frame(frame=annotated)

        frame_number += 1

    if sink:
        sink.__exit__(None, None, None)

    # Drop fragmented (ultra-short) player tracks before events.
    players, dropped = drop_short_tracks(players, min_track_frames)
    if dropped:
        print(f"Dropped {dropped} short player track(s) (< {min_track_frames} frames); "
              f"{len(players)} tracks kept.")

    # Bridge short ball-detection gaps so possession survives a few missed frames.
    filled = interpolate_ball(ball, frame_number - 1, max_gap=ball_interp_gap)
    if filled:
        print(f"Interpolated the ball across {filled} missed frame(s) "
              f"(gaps <= {ball_interp_gap}).")

    output_path = os.path.join(output_dir, video_name + ".csv")
    save_tracking_results(players, ball, frame_number, output_path)

    # Sidecar so downstream tools recover the real timeline: with a stride the
    # CSV's consecutive frame numbers tick at ``effective_fps``, not the video's
    # fps, and every timestamp would be off by the stride factor without this.
    meta_path = os.path.join(output_dir, video_name + ".meta.json")
    with open(meta_path, "w") as meta_file:
        json.dump({
            "video": os.path.basename(video_path),
            "source_fps": fps,
            "frame_stride": frame_stride,
            "effective_fps": effective_fps,
            "tracked_frames": frame_number - 1,
        }, meta_file, indent=2)
    print(f"Metadata: {meta_path} (effective_fps={effective_fps:g})")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 (ultralytics) tracking on match footage.")
    parser.add_argument("--video", required=True, help="Path to the input .mp4 footage.")
    parser.add_argument("--output", default="./track/output", help="Directory for the tracking CSV.")
    parser.add_argument("--player-model", default=DEFAULT_PLAYER_MODEL,
                        help="Weights for player detection. 'football' (default) downloads a "
                             "football-trained model from the HF Hub (fallback yolov8n.pt); "
                             "or pass a path to a .pt file.")
    parser.add_argument("--ball-model", default=DEFAULT_BALL_MODEL,
                        help="Weights for ball detection. 'football' (default) downloads a "
                             "football-trained model from the HF Hub (fallback yolov8m.pt); "
                             "or pass a path to a .pt file.")
    parser.add_argument("--ball-conf", type=float, default=DEFAULT_BALL_CONF,
                        help="Confidence threshold for ball detection (default 0.15; the ball "
                             "is small and easily missed).")
    parser.add_argument("--ball-interp-gap", type=int, default=BALL_INTERP_MAX_GAP,
                        help="Max consecutive missed frames to linearly interpolate the ball "
                             "across (default 15).")
    parser.add_argument("--track-buffer", type=int, default=DEFAULT_LOST_TRACK_BUFFER,
                        help="ByteTrack lost-track buffer in frames: how long a lost player "
                             "keeps its id before a new one is minted (default 90).")
    parser.add_argument("--min-track-frames", type=int, default=DEFAULT_MIN_TRACK_FRAMES,
                        help="Discard player tracks shorter than this many frames (default 12).")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="Inference resolution (default 1280). Higher recovers more small "
                             "ball detections but is slower; 640 is the fast/low-recall option.")
    parser.add_argument("--pitch-model", default=DEFAULT_PITCH_MODEL,
                        help="Pitch keypoint model for homography. 'football-field' (default) "
                             "downloads it from the HF Hub; 'none' disables homography and uses "
                             "image-space coords; or pass a path to a .pt file.")
    parser.add_argument("--pitch-conf", type=float, default=DEFAULT_PITCH_CONF,
                        help="Min keypoint confidence used in the homography (default 0.5).")
    parser.add_argument("--homography-every", type=int, default=DEFAULT_HOMOGRAPHY_EVERY,
                        help="Recompute the homography every N frames (default 5).")
    parser.add_argument("--pitch-imgsz", type=int, default=DEFAULT_PITCH_IMGSZ,
                        help="Resolution cap for the pitch keypoint model (default 1280); kept "
                             "lower than --imgsz since field landmarks don't need high res.")
    parser.add_argument("--no-teams", action="store_true", help="Disable team classification.")
    parser.add_argument("--generate-video", action="store_true", help="Also write an annotated video.")
    parser.add_argument("--stride", type=int, default=30, help="Frame stride for team-model crops.")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Run detection on every Nth video frame (2 = half the GPU cost). "
                             "Frame-count options are rescaled automatically so their meaning "
                             "in seconds is unchanged; the true timeline is recorded in the "
                             "sidecar .meta.json. Higher values fragment tracks more.")
    return parser.parse_args()


def main():
    args = parse_args()
    out = track(
        video_path=args.video,
        output_dir=args.output,
        player_model_path=args.player_model,
        ball_model_path=args.ball_model,
        ball_conf=args.ball_conf,
        ball_interp_gap=args.ball_interp_gap,
        track_buffer=args.track_buffer,
        min_track_frames=args.min_track_frames,
        imgsz=args.imgsz,
        pitch_model_path=args.pitch_model,
        pitch_conf=args.pitch_conf,
        homography_every=args.homography_every,
        pitch_imgsz=args.pitch_imgsz,
        track_teams=not args.no_teams,
        generate_video=args.generate_video,
        stride=args.stride,
        frame_stride=args.frame_stride,
    )
    print(f"Tracking data written to: {out}")


if __name__ == "__main__":
    main()
