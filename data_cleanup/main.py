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
import os

import numpy as np
from tqdm import tqdm

# Default ultralytics weights. ``yolov8n.pt`` downloads automatically on first
# use and needs no API key.
DEFAULT_PLAYER_MODEL = "yolov8n.pt"

# Ball model. The sentinel ``"football"`` resolves (in ``resolve_ball_model_path``)
# to a football-specific checkpoint downloaded from the Hugging Face Hub, with a
# fallback to ``yolov8m.pt``. Pass any ``.pt`` path to override.
DEFAULT_BALL_MODEL = "football"
FALLBACK_BALL_MODEL = "yolov8m.pt"

# Public, football-trained YOLOv8 weights (Roboflow "football-players-detection"
# dataset, broadcast footage; classes: ball/goalkeeper/player/referee).
# ``best.pt`` is a standard ultralytics checkpoint, downloadable without an API
# key. mAP@0.5 ~0.785. Loads directly via ``YOLO(path)``.
FOOTBALL_BALL_MODEL_REPO = "uisikdag/yolo-v8-football-players-detection"
FOOTBALL_BALL_MODEL_FILE = "best.pt"

# The ball is small and easily missed; keep its confidence low.
DEFAULT_BALL_CONF = 0.15
DEFAULT_PLAYER_CONF = 0.3

# Linearly interpolate ball position across detection gaps no longer than this
# many frames (longer gaps stay empty). Prevents possession from resetting on
# every missed detection.
BALL_INTERP_MAX_GAP = 15

# COCO class ids produced by the stock yolov8 weights.
COCO_PERSON = 0
COCO_SPORTS_BALL = 32

# Standard pitch dimensions (centimetres) used to scale normalised image-space
# coordinates. Matches the dimensions assumed by ``lib.pitch`` for "raw" data.
PITCH_LENGTH_CM = 12000.0
PITCH_WIDTH_CM = 7000.0


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


def resolve_ball_model_path(spec):
    """Resolve the ``--ball-model`` argument to a loadable weights path.

    ``"football"`` (the default) downloads a football-trained YOLOv8 checkpoint
    from the Hugging Face Hub (no API key) and returns its local path, falling
    back to ``yolov8m.pt`` if the download is unavailable. Any other value is
    returned unchanged (an explicit ``.pt`` path or an ultralytics model name).
    """
    if spec != "football":
        return spec
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=FOOTBALL_BALL_MODEL_REPO, filename=FOOTBALL_BALL_MODEL_FILE)
        print(f"Ball model: football-specific weights "
              f"{FOOTBALL_BALL_MODEL_REPO}/{FOOTBALL_BALL_MODEL_FILE}")
        return path
    except Exception as exc:  # network down, hub error, or missing huggingface_hub
        print(f"Could not fetch football ball model ({exc}); "
              f"falling back to {FALLBACK_BALL_MODEL}")
        return FALLBACK_BALL_MODEL


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


def get_detections(frame, player_result, ball_result, tracker, team_classifier,
                   frame_w, frame_h, ball_class_id=COCO_SPORTS_BALL):
    import supervision as sv

    # Players (COCO "person").
    detections = sv.Detections.from_ultralytics(player_result)
    players_detections = detections[detections.class_id == COCO_PERSON]
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

    players_detections.data["pitch_xy"] = image_to_pitch(players_detections, frame_w, frame_h)
    ball_detections.data["pitch_xy"] = image_to_pitch(ball_detections, frame_w, frame_h)

    return players_detections, ball_detections


def generate_team_model(video_path, player_model, stride=30):
    import supervision as sv
    from sports.common.team import TeamClassifier

    frame_generator = sv.get_video_frames_generator(source_path=video_path, stride=stride)
    crops = []
    for frame in tqdm(frame_generator, desc="collecting crops"):
        result = player_model(frame, conf=0.3, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id == COCO_PERSON]
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


def save_tracking_results(players, ball, frames, output_path):
    csv = "Frame,Object,Object ID,Team,X1,Y1,X2,Y2,X_Pitch,Y_Pitch\n"
    for frame in range(1, frames):
        for player_id, player_data in players.items():
            if str(frame) in player_data:
                d = player_data[str(frame)]
                csv += ",".join(map(str, [
                    frame, "player", player_id, d["Team"],
                    d["X1"], d["Y1"], d["X2"], d["Y2"], d["X_Pitch"], d["Y_Pitch"],
                ])) + "\n"
        if str(frame) in ball:
            d = ball[str(frame)]
            csv += ",".join(map(str, [
                frame, "ball", "", "",
                d["X1"], d["Y1"], d["X2"], d["Y2"], d["X_Pitch"], d["Y_Pitch"],
            ])) + "\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(csv)
    return output_path


def track(video_path, output_dir,
          player_model_path=DEFAULT_PLAYER_MODEL,
          ball_model_path=DEFAULT_BALL_MODEL,
          ball_conf=DEFAULT_BALL_CONF, ball_interp_gap=BALL_INTERP_MAX_GAP,
          track_teams=True, generate_video=False, stride=30):
    import supervision as sv
    from ultralytics import YOLO

    player_model = YOLO(player_model_path)  # downloads automatically, no API key needed

    # Resolve and load the ball model, falling back to yolov8m.pt on any failure.
    resolved_ball_path = resolve_ball_model_path(ball_model_path)
    try:
        ball_model = YOLO(resolved_ball_path)
    except Exception as exc:
        print(f"Failed to load ball model '{resolved_ball_path}' ({exc}); "
              f"using {FALLBACK_BALL_MODEL}")
        ball_model = YOLO(FALLBACK_BALL_MODEL)
    ball_class_id = resolve_ball_class_id(ball_model)

    ellipse_annotator = triangle_annotator = label_annotator = None
    if generate_video:
        ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']), thickness=2)
        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'), text_position=sv.Position.BOTTOM_CENTER)
        triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'), base=25, height=21, outline_thickness=1)

    tracker = sv.ByteTrack()
    tracker.reset()

    team_classifier = None
    if track_teams:
        team_classifier = generate_team_model(video_path, player_model, stride=stride)

    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    frame_w, frame_h = video_info.width, video_info.height
    frame_generator = sv.get_video_frames_generator(video_path)

    players, ball = {}, {}
    frame_number = 1

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    annotated_video_path = os.path.join(output_dir, video_name + "_tracked.mp4")
    sink = sv.VideoSink(target_path=annotated_video_path, video_info=video_info) if generate_video else None
    if sink:
        sink.__enter__()

    for frame in tqdm(frame_generator, desc="Collecting Tracking Data..."):
        player_result = player_model(frame, conf=DEFAULT_PLAYER_CONF, verbose=False)[0]
        ball_result = ball_model(frame, conf=ball_conf, verbose=False)[0]

        all_detections, ball_detections = get_detections(
            frame, player_result, ball_result, tracker, team_classifier, frame_w, frame_h,
            ball_class_id=ball_class_id)

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
            ball[str(frame_number)] = {
                "X1": b.xyxy[0][0], "Y1": b.xyxy[0][1], "X2": b.xyxy[0][2], "Y2": b.xyxy[0][3],
                "X_Pitch": ball_pitch_xys[0][0], "Y_Pitch": ball_pitch_xys[0][1],
            }
        else:
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

    # Bridge short ball-detection gaps so possession survives a few missed frames.
    filled = interpolate_ball(ball, frame_number - 1, max_gap=ball_interp_gap)
    if filled:
        print(f"Interpolated the ball across {filled} missed frame(s) "
              f"(gaps <= {ball_interp_gap}).")

    output_path = os.path.join(output_dir, video_name + ".csv")
    save_tracking_results(players, ball, frame_number, output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 (ultralytics) tracking on match footage.")
    parser.add_argument("--video", required=True, help="Path to the input .mp4 footage.")
    parser.add_argument("--output", default="./track/output", help="Directory for the tracking CSV.")
    parser.add_argument("--player-model", default=DEFAULT_PLAYER_MODEL,
                        help="Ultralytics weights for player detection (default yolov8n.pt).")
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
    parser.add_argument("--no-teams", action="store_true", help="Disable team classification.")
    parser.add_argument("--generate-video", action="store_true", help="Also write an annotated video.")
    parser.add_argument("--stride", type=int, default=30, help="Frame stride for team-model crops.")
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
        track_teams=not args.no_teams,
        generate_video=args.generate_video,
        stride=args.stride,
    )
    print(f"Tracking data written to: {out}")


if __name__ == "__main__":
    main()
