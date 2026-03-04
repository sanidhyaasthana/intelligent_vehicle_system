from pathlib import Path
import numpy as np
import pandas as pd


class EnrollmentDB:
    """Preload embeddings into RAM and allow lookup by user_id.

    Expects a parquet or csv with columns: user_id, embedding (list/np array)
    """

    def __init__(self, path: str, expected_backbone=None, expected_dim=None):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Enrollment DB file not found: {path}")

        if p.suffix in ['.parquet', '.pq']:
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)

        # Expect embedding columns to be stored as lists or numeric columns emb_0..emb_N
        if 'embedding' in df.columns:
            emb_col = df['embedding'].apply(lambda x: np.array(x) if not isinstance(x, (list, np.ndarray)) else np.array(x))
            df['embedding'] = emb_col
        else:
            # Try to detect columns starting with emb_
            emb_cols = [c for c in df.columns if c.startswith('emb_')]
            if emb_cols:
                df['embedding'] = df[emb_cols].values.tolist()

        # Enrollment provenance metadata
        meta = {}
        if 'backbone_name' in df.columns:
            meta['backbone_name'] = str(df['backbone_name'].iloc[0])
        elif hasattr(df, 'attrs') and 'backbone_name' in df.attrs:
            meta['backbone_name'] = str(df.attrs['backbone_name'])
        else:
            meta['backbone_name'] = None
        if 'embedding_dim' in df.columns:
            meta['embedding_dim'] = int(df['embedding_dim'].iloc[0])
        elif hasattr(df, 'attrs') and 'embedding_dim' in df.attrs:
            meta['embedding_dim'] = int(df.attrs['embedding_dim'])
        else:
            meta['embedding_dim'] = None
        self.meta = meta
        if expected_backbone is not None:
            assert meta['backbone_name'] == expected_backbone, f"Enrollment backbone mismatch: {meta['backbone_name']} != {expected_backbone}"
        if expected_dim is not None:
            assert meta['embedding_dim'] == expected_dim, f"Enrollment embedding_dim mismatch: {meta['embedding_dim']} != {expected_dim}"

        self._map = {}
        for _, row in df.iterrows():
            uid = int(row['user_id'])
            emb = np.array(row['embedding'], dtype=np.float32)
            assert emb.shape[-1] >= 1, "embedding must be a vector"
            self._map[uid] = emb

    def get(self, user_id: int):
        emb = self._map.get(int(user_id), None)
        if emb is None:
            return None
        assert emb.shape[-1] == emb.size, "Unexpected embedding shape"
        return emb

    def contains(self, user_id: int) -> bool:
        return int(user_id) in self._map
