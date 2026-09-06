import unittest
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
from data_cleanup.eval_ball import isolation_status
from events_model.wasb_sparse import SparseBallDataset, center_loss, center_heatmap, split_frames


class SupervisionTests(unittest.TestCase):
    def test_sparse_neighbors_are_unknown(self):
        self.assertEqual(isolation_status(12, {12: (10, 10)}, set(), {}), 'unknown')
        self.assertEqual(isolation_status(12, {12: (10, 10)}, {10, 11, 13, 14}, {}), 'isolated')
        self.assertEqual(isolation_status(12, {11: (10, 10)}, set(), {11: [(10, 10)]}), 'supported')

    def test_unknown_output_channels_receive_no_gradient(self):
        pred = torch.ones(2, 3, 32, 32, requires_grad=True)
        loss = center_loss(center_heatmap({0: pred}), torch.tensor([[.5, .5], [.2, .3]]), torch.tensor([True, False]))
        loss.backward()
        self.assertEqual(float(pred.grad[:, [0, 2]].abs().sum()), 0.)
        self.assertGreater(float(pred.grad[:, 1].abs().sum()), 0.)

    def test_split_has_no_shared_context(self):
        train, val = split_frames(range(1, 300), 2, 24)
        self.assertGreater((min(val) - max(train)) * 2, 5 * 24)

    def test_source_frame_mapping_and_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = str(Path(tmp) / 'tiny.avi')
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'MJPG'), 24, (64, 32))
            for i in range(12):
                writer.write(np.full((32, 64, 3), i * 20, dtype=np.uint8))
            writer.release()
            ds = SparseBallDataset(video, {4: (32., 16., 1)}, [4], 2, 64, 32)
            images, xy, visible, size, frame = ds[0]
            # CSV 4 -> source 6; context is source 5,6,7, not CSV 3,4,5.
            values = [(float(images[c].mean()) * .229 + .485) * 255 for c in (0, 3, 6)]
            np.testing.assert_allclose(values, [100, 120, 140], atol=2)
            self.assertEqual(frame, 4)
            self.assertTrue(visible)
            torch.testing.assert_close(xy, torch.tensor([.5, .5]))


if __name__ == '__main__':
    unittest.main()
