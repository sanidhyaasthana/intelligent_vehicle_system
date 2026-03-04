"""Build a simple enrollment embedding DB by extracting embeddings per identity.

This reads `data/face/labels.txt` and computes an average L2-normalized
embedding per identity using the frozen backbone. Writes a parquet with
columns: user_id, embedding
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# Ensure project root is on sys.path when executed directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _get_train_identity_ids(
    labels_path: Path,
    seed: int,
    split_ratio: tuple = (0.7, 0.15, 0.15),
    split_report: str = None,
) -> set:
    """
    Reproduce the subject-disjoint identity split used by FaceDataset and return
    only the training-partition identity IDs.

    This guarantees that enrollment embeddings are built ONLY from training
    identities — val and test identities are never enrolled.

    Raises RuntimeError if any identity in test_ids also appears in train_ids
    (sanity check against contamination).
    """
    import json as _json
    import sys as _sys

    df = pd.read_csv(labels_path)
    if 'identity_id' not in df.columns:
        raise RuntimeError(
            f"labels file {labels_path} must contain an 'identity_id' column."
        )

    # Use np.random.default_rng for reproducibility (same as make_subject_disjoint_split)
    unique_ids = sorted(df['identity_id'].unique().tolist())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    n_ids = len(shuffled)
    n_train = int(n_ids * split_ratio[0])
    n_val = int(n_ids * split_ratio[1])

    train_ids = set(shuffled[:n_train].tolist())
    val_ids   = set(shuffled[n_train:n_train + n_val].tolist())
    test_ids  = set(shuffled[n_train + n_val:].tolist())

    # --- Contamination guard ---
    for a_name, a_set, b_name, b_set in [
        ('train', train_ids, 'test', test_ids),
        ('train', train_ids, 'val', val_ids),
        ('val', val_ids, 'test', test_ids),
    ]:
        overlap = a_set & b_set
        if overlap:
            raise RuntimeError(
                f"Enrollment contamination: {len(overlap)} identities in both "
                f"{a_name} and {b_name}. Identity leakage detected."
            )

    print(
        f"[build_embedding_db] Identity split (seed={seed}): "
        f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )
    print(
        f"[build_embedding_db] Enrolling {len(train_ids)} training identities only. "
        f"Val/test identities excluded from enrollment DB."
    )

    # Save split report if requested
    if split_report is not None:
        Path(split_report).parent.mkdir(parents=True, exist_ok=True)
        report = {
            'seed': seed,
            'id_col': 'identity_id',
            'split_ratio': list(split_ratio),
            'total_identities': n_ids,
            'train_identities': len(train_ids),
            'val_identities': len(val_ids),
            'test_identities': len(test_ids),
            'train_rows': int((df['identity_id'].isin(train_ids)).sum()),
            'val_rows': int((df['identity_id'].isin(val_ids)).sum()),
            'test_rows': int((df['identity_id'].isin(test_ids)).sum()),
        }
        with open(split_report, 'w') as f:
            _json.dump(report, f, indent=2)
        print(f"[build_embedding_db] Saved split report to {split_report}")

    return train_ids


def main(out_path: str, batch_size: int = 32, image_size: int = 112,
         seed: int = 42, split_ratio: tuple = (0.7, 0.15, 0.15),
         split_report: str = None):

    import random
    import torch
    # Set deterministic seed for reproducibility
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from src.models.embedding_extractor import EmbeddingExtractor
    except Exception as e:
        raise RuntimeError(f"Required modules not available: {e}")

    import torch
    device = torch.device('cpu')

    extractor = EmbeddingExtractor(backbone_name='resnet50', embedding_dim=512, device=device)

    labels_path = Path('data/face/labels.txt')
    if not labels_path.exists():
        raise RuntimeError('data/face/labels.txt not found')

    df = pd.read_csv(labels_path)
    rows = list(df.itertuples(index=False, name=None))
    if len(rows) == 0:
        raise RuntimeError('labels.txt is empty')

    # -----------------------------------------------------------------
    # ENROLLMENT ISOLATION: build enrollment DB only from training
    # identities. Val and test identities are explicitly excluded.
    # -----------------------------------------------------------------
    train_ids = _get_train_identity_ids(labels_path, seed=SEED, split_ratio=split_ratio,
                                        split_report=split_report)

    # Filter rows to training identities only
    rows_train = [r for r in rows if int(r[1]) in train_ids]
    n_excluded = len(rows) - len(rows_train)
    print(
        f"[build_embedding_db] Total rows: {len(rows)}, "
        f"training-identity rows: {len(rows_train)}, "
        f"excluded (val/test): {n_excluded}"
    )
    rows = rows_train

    id_to_embs = {}

    def preprocess_image(img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f'Failed to read image {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size)).astype('float32') / 255.0
        img_t = np.transpose(img, (2, 0, 1))
        return img_t

    batch_imgs = []
    batch_uids = []
    with torch.no_grad():
        for img_rel, uid in rows:
            img_path = Path('data') / 'face' / img_rel
            img_t = preprocess_image(img_path)
            batch_imgs.append(img_t)
            batch_uids.append(int(uid))

            if len(batch_imgs) >= batch_size:
                batch_np = np.stack(batch_imgs, axis=0)
                batch_tensor = torch.from_numpy(batch_np).float()
                embs = extractor.extract(batch_tensor).cpu().numpy()
                for u, e in zip(batch_uids, embs):
                    id_to_embs.setdefault(int(u), []).append(e)
                batch_imgs = []
                batch_uids = []

        # flush
        if len(batch_imgs) > 0:
            batch_np = np.stack(batch_imgs, axis=0)
            batch_tensor = torch.from_numpy(batch_np).float()
            embs = extractor.extract(batch_tensor).cpu().numpy()
            for u, e in zip(batch_uids, embs):
                id_to_embs.setdefault(int(u), []).append(e)

    out_rows = []
    for uid, emb_list in id_to_embs.items():
        emb_avg = np.mean(np.stack(emb_list, axis=0), axis=0)
        out_rows.append({'user_id': int(uid), 'embedding': emb_avg.tolist()})

    out_df = pd.DataFrame(out_rows)
    out_df.to_parquet(out, index=False)
    print(f'Wrote enrollment DB to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=False, default='data/enrollment_embeddings.parquet')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--image-size', type=int, default=112)
    p.add_argument('--seed', type=int, default=42,
                   help='Seed used in FaceDataset split — must match training seed')
    p.add_argument('--split-report', default=None,
                   help='Path to save results/split_report.json')
    args = p.parse_args()
    main(args.out, batch_size=args.batch_size, image_size=args.image_size,
         seed=args.seed, split_report=args.split_report)
