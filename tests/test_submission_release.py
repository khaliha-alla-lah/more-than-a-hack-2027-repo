"""Regression checks around release reliability and trusted model loading."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from possession.submission import unreliable_ball, load_models


class SubmissionReleaseTests(unittest.TestCase):
    def test_halftime_coordinate_change_does_not_trigger_warning(self):
        m = SimpleNamespace(n_frames=100, fps=25, halves=[(0,50),(50,100)],
                            ball_x=np.r_[np.zeros(50),np.full(50,100.)], ball_y=np.zeros(100))
        f = {'shared': {'ball_known': np.ones(100)}}
        self.assertFalse(unreliable_ball(m, f).any())

    def test_finite_but_teleporting_ball_triggers_warning(self):
        x = np.zeros(100); x[50] = 20
        m = SimpleNamespace(n_frames=100, fps=25, halves=[(0,100)], ball_x=x, ball_y=np.zeros(100))
        self.assertTrue(unreliable_ball(m, {'shared': {'ball_known': np.ones(100)}})[50])

    def test_checkpoint_tampering_fails_before_deserialization(self):
        with TemporaryDirectory() as folder:
            p = Path(folder)
            (p/'manifest.json').write_text(json.dumps({'models': {'baseline.pkl': '0'*64}}))
            (p/'baseline.pkl').write_bytes(b'not a pickle')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_models(p)


if __name__ == '__main__':
    unittest.main()
