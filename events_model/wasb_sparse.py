"""Fine-tune WASB using real center labels and unlabeled temporal context.

No XML conversion, interpolated targets, or supervision of unknown neighbors.
See WASB_SPARSE.md for protocol and commands. Video reads stay at source resolution
until resizing; CSV frame k maps to source frame (k-1)*frame_stride.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader


def read_labels(path):
    labels = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            k, visible = int(row['Frame']), int(row['visible'])
            if k < 1 or visible not in (0, 1) or k in labels:
                raise ValueError(f'Invalid or duplicate label at frame {k}')
            xy = (float(row['X_img']), float(row['Y_img'])) if visible else (0., 0.)
            if not all(math.isfinite(v) and v >= 0 for v in xy):
                raise ValueError(f'Invalid coordinates at frame {k}')
            labels[k] = (*xy, visible)
    if not labels:
        raise ValueError('Empty labels')
    return labels


def split_frames(frames, stride, fps, val_fraction=.3, embargo_seconds=5.):
    """Chronological holdout; no source images shared across the boundary."""
    frames = sorted(frames)
    cut = max(1, int(len(frames) * (1 - val_fraction)))
    if cut >= len(frames):
        raise ValueError('Not enough labels for train/validation')
    boundary = frames[cut]
    gap = max(3, math.ceil(embargo_seconds * fps / stride))
    train = [f for f in frames if f < boundary - gap]
    val = frames[cut:]
    if not train or not val:
        raise ValueError('Empty split after temporal embargo')
    return train, val


class SparseBallDataset(Dataset):
    def __init__(self, video, labels, frames, stride, width=512, height=288):
        self.video, self.labels, self.frames = str(video), labels, frames
        self.stride, self.width, self.height = stride, width, height

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        k = self.frames[index]
        center = (k - 1) * self.stride
        cap = cv2.VideoCapture(self.video)
        images = []
        try:
            # Adjacent source frames preserve full temporal resolution.
            for source in (center - 1, center, center + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, source)
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f'Cannot read source frame {source}: {self.video}')
                h, w = bgr.shape[:2]
                rgb = cv2.cvtColor(cv2.resize(bgr, (self.width, self.height)), cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
                image = (image - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
                images.append(image)
        finally:
            cap.release()
        x, y, visible = self.labels[k]
        if visible and (x >= w or y >= h):
            raise ValueError(f'Label {k} outside video dimensions {w}x{h}')
        return torch.cat(images), torch.tensor([x / w, y / h]), bool(visible), torch.tensor([w, h]), k


def center_heatmap(output):
    maps = [output] if torch.is_tensor(output) else list(output.values())
    pred = max(maps, key=lambda t: t.shape[-2] * t.shape[-1])
    if pred.ndim != 4 or pred.shape[1] != 3:
        raise ValueError(f'Expected WASB B,3,H,W output, got {pred.shape}')
    return pred[:, 1]  # no gradient through unlabeled neighbor outputs


def center_loss(pred, xy, visible, sigma=2.):
    """Balanced foreground/background MSE, only on genuinely annotated centers."""
    h, w = pred.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=pred.device),
                            torch.arange(w, device=pred.device), indexing='ij')
    target = torch.exp(-((xx - xy[:, 0, None, None] * w)**2 +
                         (yy - xy[:, 1, None, None] * h)**2) / (2 * sigma**2))
    target = target * visible[:, None, None]
    error = (pred - target)**2
    fg = (error * target).sum((1, 2)) / target.sum((1, 2)).clamp_min(1)
    bg = (error * (1 - target)).sum((1, 2)) / (1 - target).sum((1, 2)).clamp_min(1)
    return (fg + bg).mean()


@torch.no_grad()
def evaluate(model, loader, device, threshold):
    model.eval()
    distances, predicted_distances, invisible_picks = [], [], 0
    n_invisible = 0
    rows = []
    for images, xy, visible, size, frames in loader:
        hm = center_heatmap(model(images.to(device))).cpu()
        h, w = hm.shape[-2:]
        scores, indices = hm.flatten(1).max(1)
        positions = torch.stack((indices % w / w, indices // w / h), 1) * size
        for i, frame in enumerate(frames):
            picked = float(scores[i]) >= threshold
            distance = float(torch.linalg.vector_norm(positions[i] - xy[i] * size[i]))
            if visible[i]:
                distances.append(distance if picked else float('inf'))
                if picked:
                    predicted_distances.append(distance)
            else:
                n_invisible += 1
                invisible_picks += picked
            rows.append({'Frame': int(frame), 'X_img': float(positions[i, 0]),
                         'Y_img': float(positions[i, 1]), 'score': float(scores[i]),
                         'predicted_visible': int(picked)})
    if not distances:
        raise ValueError('Validation has no visible labels')
    metrics = {f'acc@{r}': sum(d <= r for d in distances) / len(distances) for r in (20, 50, 100)}
    metrics.update(visible=len(distances), invisible=n_invisible,
                   precision_visible=sum(d <= 100 for d in predicted_distances) / max(1, len(predicted_distances)),
                   false_positive_invisible=invisible_picks / max(1, n_invisible),
                   threshold=threshold)
    return metrics, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--meta', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--wasb-src', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--height', type=int, default=288)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if min(args.width, args.height) < 32 or args.width % 32 or args.height % 32:
        p.error('width and height must be positive multiples of 32')
    if args.out.exists():
        p.error('Output already exists; use a new experiment directory')
    labels = read_labels(args.labels)
    meta = json.loads(args.meta.read_text())
    stride, fps = int(meta['frame_stride']), float(meta['source_fps'])
    if stride < 1 or fps <= 0:
        p.error('Invalid metadata')
    cap = cv2.VideoCapture(str(args.video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not count or abs(actual_fps - fps) > .1:
        p.error('Video unavailable or FPS does not match metadata')
    if any((f - 1) * stride >= count for f in labels):
        p.error('Labels exceed video length; check video and stride')
    frames = [f for f in labels if 0 < (f - 1) * stride < count - 1]
    train, val = split_frames(frames, stride, fps)
    report = dict(train_frames=train, validation_frames=val,
                  excluded_frames=sorted(set(labels) - set(train) - set(val)),
                  train_visible=sum(labels[f][2] for f in train),
                  validation_visible=sum(labels[f][2] for f in val),
                  interpolated_targets=0, unknown_negative_targets=0,
                  source_fps=fps, frame_stride=stride,
                  settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    if not report['train_visible'] or not report['validation_visible']:
        p.error('Both splits need visible labels')
    args.out.mkdir(parents=True)
    (args.out / 'split.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('_frames')}, indent=2))
    if args.audit_only:
        return
    if not args.wasb_src or not args.checkpoint:
        p.error('Training requires --wasb-src and --checkpoint')
    import sys
    sys.path.insert(0, str(args.wasb_src.resolve()))
    from hydra import compose, initialize_config_dir
    from models import build_model
    with initialize_config_dir(version_base=None, config_dir=str((args.wasb_src / 'configs').resolve())):
        cfg = compose(config_name='eval', overrides=['dataset=soccer', 'model=wasb'])
    torch.manual_seed(42)
    model = build_model(cfg).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    loaders = [DataLoader(SparseBallDataset(args.video, labels, fs, stride, args.width, args.height),
                          batch_size=args.batch_size, shuffle=shuffle, num_workers=0)
               for fs, shuffle in ((train, True), (val, False))]
    baseline, rows = evaluate(model, loaders[1], args.device, args.threshold)
    history = [dict(epoch=-1, **baseline)]
    best = baseline['acc@100']
    torch.save({'model_state_dict': model.state_dict(), 'epoch': -1, 'metrics': baseline}, args.out / 'best.pth')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('BEFORE', baseline, flush=True)
    for epoch in range(args.epochs):
        model.train()
        # Tiny batches otherwise destroy pretrained batch-normalization statistics.
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        for images, xy, visible, _, _ in loaders[0]:
            prediction = center_heatmap(model(images.to(args.device)))
            loss = center_loss(prediction, xy.to(args.device), visible.to(args.device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        metrics, predictions = evaluate(model, loaders[1], args.device, args.threshold)
        history.append(dict(epoch=epoch, **metrics))
        print(epoch, metrics, flush=True)
        if metrics['acc@100'] > best:
            best, rows = metrics['acc@100'], predictions
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'metrics': metrics}, args.out / 'best.pth')
        (args.out / 'metrics.json').write_text(json.dumps(history, indent=2))
    (args.out / 'metrics.json').write_text(json.dumps(history, indent=2))
    with (args.out / 'validation_predictions.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
