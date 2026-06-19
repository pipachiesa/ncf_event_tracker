"""
Command-line tracking entry point.

This is a script version of ``track/track.ipynb``: it runs the Roboflow player
and pitch detection models over a video and writes raw tracking data (player &
ball pitch coordinates per frame) to a CSV that the rest of the pipeline
(``data_cleanup`` cleaning and ``event_generation``) can consume.

Example:
    python data_cleanup/main.py \\
        --video ./track/footage/2e57b9_0.mp4 \\
        --output ./track/output \\
        --api-key $ROBOFLOW_API

The CSV is written to ``<output>/<video name>.csv`` in the "raw" format read by
``Match.import_raw_data``.
"""

import argparse
import os

import numpy as np
from tqdm import tqdm

# Public Roboflow Universe models used by the tracking notebook. Override with
# the project/version you trained yourself via --player-model / --pitch-model.
DEFAULT_PLAYER_MODEL_ID = "football-players-detection-3zvbc-btky1/1"
DEFAULT_PITCH_MODEL_ID = "football-field-detection-f07vi-yukgc/1"

OBJECTS = {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}


def resolve_goalkeepers_team_id(players, goalkeepers):
    import supervision as sv

    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def get_detections(frame, detections, key_points, tracker, team_classifier):
    import supervision as sv
    from sports.common.view import ViewTransformer
    from sports.configs.soccer import SoccerPitchConfiguration

    CONFIG = SoccerPitchConfiguration()

    ball_detections = detections[detections.class_id == OBJECTS["ball"]]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    all_detections = detections[detections.class_id != OBJECTS["ball"]]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    all_detections = tracker.update_with_detections(detections=all_detections)

    goalkeepers_detections = all_detections[all_detections.class_id == OBJECTS["goalkeeper"]]
    players_detections = all_detections[all_detections.class_id == OBJECTS["player"]]

    if team_classifier:
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)

    goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
        players_detections, goalkeepers_detections)

    # Project image-space detections onto the 2D pitch.
    confidence_filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][confidence_filter]
    pitch_reference_points = np.array(CONFIG.vertices)[confidence_filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points,
    )

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    ball_detections.data["pitch_xy"] = transformer.transform_points(points=frame_ball_xy)

    frame_goalkeepers_xy = goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    goalkeepers_detections.data["pitch_xy"] = transformer.transform_points(points=frame_goalkeepers_xy)

    frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_detections.data["pitch_xy"] = transformer.transform_points(points=frame_players_xy)

    all_detections = sv.Detections.merge([players_detections, goalkeepers_detections])
    return all_detections, ball_detections


def generate_team_model(video_path, player_model, stride=30):
    import supervision as sv
    from sports.common.team import TeamClassifier

    frame_generator = sv.get_video_frames_generator(source_path=video_path, stride=stride)
    crops = []
    for frame in tqdm(frame_generator, desc="collecting crops"):
        result = player_model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
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


def track(video_path, output_dir, api_key,
          player_model_id=DEFAULT_PLAYER_MODEL_ID,
          pitch_model_id=DEFAULT_PITCH_MODEL_ID,
          track_teams=True, generate_video=False, stride=30):
    os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", "[CUDAExecutionProvider]")

    import supervision as sv
    from inference import get_model
    from sports.configs.soccer import SoccerPitchConfiguration

    CONFIG = SoccerPitchConfiguration()

    player_model = get_model(model_id=player_model_id, api_key=api_key)
    pitch_model = get_model(model_id=pitch_model_id, api_key=api_key)

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
    frame_generator = sv.get_video_frames_generator(video_path)

    players, ball = {}, {}
    frame_number = 1

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    annotated_video_path = os.path.join(output_dir, video_name + "_tracked.mp4")
    sink = sv.VideoSink(target_path=annotated_video_path, video_info=video_info) if generate_video else None
    if sink:
        sink.__enter__()

    for frame in tqdm(frame_generator, desc="Collecting Tracking Data..."):
        result = player_model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        result = pitch_model.infer(frame, confidence=0.3)[0]
        key_points = sv.KeyPoints.from_inference(result)

        all_detections, ball_detections = get_detections(
            frame, detections, key_points, tracker, team_classifier)

        object_ids = all_detections.tracker_id
        team_ids = all_detections.class_id
        object_types = all_detections.data["class_name"]
        pitch_xys = all_detections.data["pitch_xy"]
        ball_pitch_xys = ball_detections.data["pitch_xy"]
        all_detections.class_id = all_detections.class_id.astype(int)

        for idx, xyxy in enumerate(all_detections.xyxy):
            object_id = str(object_ids[idx])
            players.setdefault(object_id, {})
            players[object_id][str(frame_number)] = {
                "Object Type": object_types[idx],
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

    output_path = os.path.join(output_dir, video_name + ".csv")
    save_tracking_results(players, ball, frame_number, output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO/Roboflow tracking on match footage.")
    parser.add_argument("--video", required=True, help="Path to the input .mp4 footage.")
    parser.add_argument("--output", default="./track/output", help="Directory for the tracking CSV.")
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API"),
                        help="Roboflow API key (defaults to the ROBOFLOW_API env var).")
    parser.add_argument("--player-model", default=DEFAULT_PLAYER_MODEL_ID)
    parser.add_argument("--pitch-model", default=DEFAULT_PITCH_MODEL_ID)
    parser.add_argument("--no-teams", action="store_true", help="Disable team classification.")
    parser.add_argument("--generate-video", action="store_true", help="Also write an annotated video.")
    parser.add_argument("--stride", type=int, default=30, help="Frame stride for team-model crops.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("A Roboflow API key is required (pass --api-key or set ROBOFLOW_API).")
    out = track(
        video_path=args.video,
        output_dir=args.output,
        api_key=args.api_key,
        player_model_id=args.player_model,
        pitch_model_id=args.pitch_model,
        track_teams=not args.no_teams,
        generate_video=args.generate_video,
        stride=args.stride,
    )
    print(f"Tracking data written to: {out}")


if __name__ == "__main__":
    main()
