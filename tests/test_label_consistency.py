"""
tests/test_label_consistency.py

Unit tests to verify label and score conventions are consistent
across dataset loading, simulation, and fusion pipeline.

Label convention: 1 = genuine, 0 = impostor
Score convention: higher = more genuine
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestLabelConvention(unittest.TestCase):
    """Verify label values and score-label correlation in event CSVs."""

    GEO_CSV = 'data/geo/system_events.csv'
    SYS_CSV = 'data/system_events/system_events.csv'

    def _load(self, path):
        p = Path(path)
        if not p.exists():
            self.skipTest(f"{path} not found")
        return pd.read_csv(p)

    def test_geo_csv_labels_binary(self):
        """Labels must be 0 or 1 only."""
        df = self._load(self.GEO_CSV)
        unique = set(df['label'].unique())
        self.assertTrue(
            unique.issubset({0, 1}),
            f"Labels must be {{0,1}}, got {unique}"
        )

    def test_geo_csv_both_classes_present(self):
        """Both genuine (1) and impostor (0) must be present."""
        df = self._load(self.GEO_CSV)
        self.assertIn(1, df['label'].values, "No genuine samples (label=1) found")
        self.assertIn(0, df['label'].values, "No impostor samples (label=0) found")

    def test_geo_csv_tface_higher_for_genuine(self):
        """T_face mean for genuine should be higher than for impostor."""
        df = self._load(self.GEO_CSV)
        gen_mean = df.loc[df['label'] == 1, 'T_face'].mean()
        imp_mean = df.loc[df['label'] == 0, 'T_face'].mean()
        self.assertGreater(
            gen_mean, imp_mean,
            f"T_face genuine_mean={gen_mean:.4f} should > impostor_mean={imp_mean:.4f}"
        )

    def test_geo_csv_tgeo_higher_for_genuine(self):
        """T_geo mean for genuine should be higher than for impostor."""
        df = self._load(self.GEO_CSV)
        gen_mean = df.loc[df['label'] == 1, 'T_geo'].mean()
        imp_mean = df.loc[df['label'] == 0, 'T_geo'].mean()
        self.assertGreater(
            gen_mean, imp_mean,
            f"T_geo genuine_mean={gen_mean:.4f} should > impostor_mean={imp_mean:.4f}"
        )

    def test_geo_csv_fusion_correlation_positive(self):
        """Fusion score must be positively correlated with labels."""
        df = self._load(self.GEO_CSV)
        fusion = 0.6 * df['T_face'] + 0.4 * df['T_geo']
        corr = float(np.corrcoef(fusion.values, df['label'].values)[0, 1])
        self.assertGreater(
            corr, 0,
            f"Fusion-label Pearson correlation={corr:.4f} must be > 0. "
            f"Score-label orientation is inverted."
        )

    def test_sys_events_labels_binary(self):
        """Raw system events labels must be 0 or 1."""
        df = self._load(self.SYS_CSV)
        unique = set(df['label'].unique())
        self.assertTrue(
            unique.issubset({0, 1}),
            f"Labels must be {{0,1}}, got {unique}"
        )

    def test_subject_disjoint_split_no_leakage(self):
        """Subject-disjoint split must produce zero overlap across partitions."""
        df = self._load(self.GEO_CSV)
        id_col = None
        for c in ('identity', 'user_id', 'identity_id', 'subject_id'):
            if c in df.columns:
                id_col = c
                break
        if id_col is None:
            self.skipTest("No identity column found for split test")

        from src.datasets.system_event_dataset import make_subject_disjoint_split
        train_ids, val_ids, test_ids = make_subject_disjoint_split(
            df, id_col=id_col, seed=42
        )

        t = set(train_ids.tolist())
        v = set(val_ids.tolist())
        te = set(test_ids.tolist())

        self.assertEqual(len(t & te), 0, "Train/test identity overlap!")
        self.assertEqual(len(t & v), 0, "Train/val identity overlap!")
        self.assertEqual(len(v & te), 0, "Val/test identity overlap!")

    def test_split_covers_all_identities(self):
        """All identities must be assigned to exactly one partition."""
        df = self._load(self.GEO_CSV)
        id_col = None
        for c in ('identity', 'user_id', 'identity_id', 'subject_id'):
            if c in df.columns:
                id_col = c
                break
        if id_col is None:
            self.skipTest("No identity column found")

        from src.datasets.system_event_dataset import make_subject_disjoint_split
        train_ids, val_ids, test_ids = make_subject_disjoint_split(
            df, id_col=id_col, seed=42
        )

        all_assigned = set(train_ids.tolist()) | set(val_ids.tolist()) | set(test_ids.tolist())
        all_ids = set(df[id_col].unique().tolist())
        self.assertEqual(
            all_assigned, all_ids,
            f"Not all identities assigned: missing {all_ids - all_assigned}"
        )


class TestTARInterpolation(unittest.TestCase):
    """Test TAR@FAR interpolation correctness."""

    def _make_roc(self, n_pos=500, n_neg=5000, seed=42):
        """Construct synthetic ROC for testing."""
        rng = np.random.default_rng(seed)
        pos_scores = rng.normal(0.7, 0.15, n_pos)
        neg_scores = rng.normal(0.3, 0.15, n_neg)
        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        return fpr, tpr, n_neg

    def test_tar_interpolation_within_range(self):
        """np.interp should produce TAR in [0, 1] for supported FAR."""
        fpr, tpr, n_neg = self._make_roc()
        target_far = 1e-2
        min_far = 1.0 / n_neg
        self.assertLess(min_far, target_far, "Test setup error: min_far >= target_far")
        tar = float(np.interp(target_far, fpr, tpr))
        self.assertGreaterEqual(tar, 0.0)
        self.assertLessEqual(tar, 1.0)

    def test_tar_unsupported_below_min_far(self):
        """TAR@1e-3 should be unsupported when n_impostors < 1000."""
        fpr, tpr, _ = self._make_roc(n_neg=500)  # 500 < 1000
        n_impostors = 500
        target_far = 1e-3
        min_far = 1.0 / n_impostors
        self.assertGreater(min_far, target_far,
                           "With 500 impostors, min_FAR=0.002 > 0.001")

    def test_tar_1e3_supported_with_10k_impostors(self):
        """With 10k impostors, TAR@1e-3 and TAR@1e-2 should both be supported."""
        n_neg = 10_000
        fpr, tpr, _ = self._make_roc(n_neg=n_neg)
        min_far = 1.0 / n_neg  # 1e-4 < 1e-3
        self.assertLess(min_far, 1e-3)
        self.assertLess(min_far, 1e-2)
        # Check ROC has points below 1e-3
        self.assertTrue(any(fpr < 1e-3), "ROC must have points below 1e-3")
        tar_1e3 = float(np.interp(1e-3, fpr, tpr))
        self.assertGreaterEqual(tar_1e3, 0.0)
        self.assertLessEqual(tar_1e3, 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
