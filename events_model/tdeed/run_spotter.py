#!/usr/bin/env python3
"""
Wrapper around inference.py: runs T-DEED on a video, saves the raw output to a
uniquely-named JSON (with video/model/threshold/fps metadata) and prints only
the Goal/Shot/Foul detections with their timestamp.

Usage:
    python run_spotter.py --model SoccerNetBall_challenge1 \
        --video /path/to/clip.mp4 --threshold 0.3
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys

import cv2

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_RESULT = os.path.join(REPO_DIR, 'inference_output', 'results_inference.json')

# Substrings that identify the events we care about, lowercase. Covers both
# label sets: soccernetball (SHOT, GOAL) and soccernet 17 classes
# (Goal, Shots on/off target, Foul).
KEY_PATTERNS = ('goal', 'shot', 'foul')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Model name, e.g. SoccerNetBall_challenge1 or SoccerNet_small')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--frame_width', type=int, default=796)
    parser.add_argument('--frame_height', type=int, default=448)
    parser.add_argument('--out_dir', type=str, default=os.path.join(REPO_DIR, 'spotter_results'))
    return parser.parse_args()


def video_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        raise RuntimeError(f'No pude leer el fps de {path}')
    return fps


def fmt_ts(seconds):
    m, s = divmod(seconds, 60)
    return f'{int(m):02d}:{s:04.1f}'


def main():
    args = get_args()
    video = os.path.abspath(args.video)
    if not os.path.exists(video):
        sys.exit(f'No existe el video: {video}')

    fps = video_fps(video)

    # inference.py always writes to the same path; remove stale output first
    if os.path.exists(RAW_RESULT):
        os.remove(RAW_RESULT)

    cmd = [sys.executable, 'inference.py',
           '--model', args.model,
           '--video_path', video,
           '--frame_width', str(args.frame_width),
           '--frame_height', str(args.frame_height),
           '--inference_threshold', str(args.threshold)]
    print('>>', ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_DIR)
    if proc.returncode != 0:
        sys.exit(f'inference.py falló (exit {proc.returncode})')
    if not os.path.exists(RAW_RESULT):
        sys.exit(f'inference.py terminó pero no generó {RAW_RESULT}')

    with open(RAW_RESULT) as fp:
        raw = json.load(fp)

    result = {
        'video': video,
        'model': args.model,
        'threshold': args.threshold,
        'fps': fps,
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'predictions': raw.get('predictions', []),
    }

    stem = os.path.splitext(os.path.basename(video))[0]
    safe = lambda s: re.sub(r'[^A-Za-z0-9._-]+', '_', s)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out_name = f'{safe(stem)}__{safe(args.model)}__thr{args.threshold}__{stamp}.json'
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, out_name)
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=4)

    key_events = [p for p in result['predictions']
                  if any(k in p['label'].lower() for k in KEY_PATTERNS)]
    key_events.sort(key=lambda p: p['frame'])

    print(f'\nResultado completo guardado en: {out_path}')
    print(f'Video: {stem} | Modelo: {args.model} | Umbral: {args.threshold} | fps: {fps:.2f}')
    if not key_events:
        print('Sin detecciones de Goal/Shot/Foul.')
    else:
        print(f'{len(key_events)} detecciones de Goal/Shot/Foul:')
        for p in key_events:
            ts = fmt_ts(p['frame'] / fps)
            print(f"  {ts}  frame {p['frame']:>6}  {p['label']:<20} conf={p['confidence']:.3f}")


if __name__ == '__main__':
    main()
