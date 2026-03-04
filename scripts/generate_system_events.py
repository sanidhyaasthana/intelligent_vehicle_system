"""Generate a compliant system_events CSV for evaluation.

Outputs CSV with columns: image_path,user_id,vehicle_location,user_location,label
 - image_path: full path to image file (relative to repo root)
 - user_id: integer identity id (matches enrollment DB user_id)
 - vehicle_location, user_location: "lat;lon" strings
 - label: 1 for genuine (same user), 0 for impostor

This script samples from data/face/labels.txt and data/geo/location_data.csv
to produce a realistic mix of genuine and impostor events.
"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path
import math

import pandas as pd

# Ensure project root is on sys.path when executed directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def haversine_deg(a, b):
    # approximate distance in degrees (Euclidean on lat/lon) used by evaluation runner
    return math.hypot(a[0] - b[0], a[1] - b[1])


def format_loc(lat, lon):
    return f"{lat};{lon}"


def main(out_path: str, n_events: int = 4000, genuine_frac: float = 0.4, seed: int = 42):
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    random.seed(seed)
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    labels_path = Path('data/face/labels.txt')
    if not labels_path.exists():
        raise RuntimeError('data/face/labels.txt not found in workspace')

    loc_path = Path('data/geo/location_data.csv')
    if not loc_path.exists():
        raise RuntimeError('data/geo/location_data.csv not found in workspace')

    # Load available image lines
    df_labels = pd.read_csv(labels_path)
    df_loc = pd.read_csv(loc_path)

    # Prepare list of (image_path, user_id)
    entries = list(df_labels.itertuples(index=False, name=None))
    if len(entries) == 0:
        raise RuntimeError('No entries in labels.txt')

    # Locations pool
    locs = df_loc[['lat', 'lon']].values.tolist()
    if len(locs) == 0:
        raise RuntimeError('No locations available in location_data.csv')

    rows = []
    n_genuine = int(n_events * genuine_frac)
    n_impostor = n_events - n_genuine

    # Helper to pick a random location
    def pick_loc():
        lat, lon = random.choice(locs)
        return float(lat), float(lon)

    # Create genuine events: vehicle_location near user_location
    for _ in range(n_genuine):
        img_rel, uid = random.choice(entries)
        img_path = os.path.join('data', 'face', img_rel)
        user_loc = pick_loc()
        # small perturbation for vehicle (simulate GPS noise)
        veh_lat = user_loc[0] + random.gauss(0, 1e-4)
        veh_lon = user_loc[1] + random.gauss(0, 1e-4)
        rows.append((img_path, int(uid), format_loc(veh_lat, veh_lon), format_loc(user_loc[0], user_loc[1]), 1))

    # Create impostor events: vehicle_location far from user_location
    for _ in range(n_impostor):
        img_rel, uid = random.choice(entries)
        img_path = os.path.join('data', 'face', img_rel)
        user_loc = pick_loc()
        # pick a vehicle location sufficiently far
        veh_loc = pick_loc()
        tries = 0
        while haversine_deg(user_loc, veh_loc) < 0.01 and tries < 50:
            veh_loc = pick_loc()
            tries += 1
        rows.append((img_path, int(uid), format_loc(veh_loc[0], veh_loc[1]), format_loc(user_loc[0], user_loc[1]), 0))

    # Shuffle rows
    random.shuffle(rows)

    # Write CSV header
    with open(outp, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'user_id', 'vehicle_location', 'user_location', 'label'])
        for r in rows:
            writer.writerow(r)

    logger.info(f'Wrote {len(rows)} system events to {outp}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=False, default='data/system_events/system_events.csv', help='Output CSV path')
    p.add_argument('--n', type=int, default=4000, help='Number of events to generate')
    p.add_argument('--genuine-frac', type=float, default=0.4, help='Fraction of genuine events')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    main(args.out, n_events=args.n, genuine_frac=args.genuine_frac, seed=args.seed)
"""Generate a compliant system_events.csv for evaluation.

Creates CSV with columns: image_path,user_id,vehicle_location,user_location,label

Labels: 1=genuine (user located near vehicle), 0=impostor (user far)
"""
import argparse
from pathlib import Path
import random
import pandas as pd
import numpy as np
from src.datasets.face_dataset import FaceDataset


def main(out_csv: str, face_image_dir: str, face_label_file: str, geo_locations_csv: str, n_samples: int = 500):
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build face dataset test split
    dataset = FaceDataset(image_dir=face_image_dir, label_file=face_label_file, split='test', precompute_quality=False)

    # Load geo locations
    geo_df = pd.read_csv(geo_locations_csv)
    locs = geo_df[['lat', 'lon']].values if {'lat','lon'}.issubset(set(geo_df.columns)) else None
    if locs is None:
        # Fallback: create random lat/lon pairs
        locs = np.random.uniform(low=-180, high=180, size=(100,2))

    rows = []
    rng = random.Random(42)
    N = min(len(dataset), n_samples)
    for i in range(N):
        sample = dataset[i]
        image_path = sample['path']
        user_id = int(sample['identity'])

        # Choose vehicle location randomly
        v_idx = rng.randrange(len(locs))
        vehicle_loc = tuple(map(float, locs[v_idx]))

        # Make genuine event with small perturbation (80% of samples) else impostor
        if rng.random() < 0.8:
            # genuine: user location near vehicle
            noise = np.random.normal(scale=0.0001, size=2)
            user_loc = (float(vehicle_loc[0] + noise[0]), float(vehicle_loc[1] + noise[1]))
            label = 1
        else:
            # impostor: choose distant location
            other = locs[rng.randrange(len(locs))]
            user_loc = (float(other[0]), float(other[1]))
            label = 0

        rows.append({
            'image_path': image_path,
            'user_id': user_id,
            'vehicle_location': f"{vehicle_loc[0]};{vehicle_loc[1]}",
            'user_location': f"{user_loc[0]};{user_loc[1]}",
            'label': int(label),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f'Wrote system events to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--face_dir', default='data/face/lfw-deepfunneled')
    p.add_argument('--labels', default='data/face/labels.txt')
    p.add_argument('--geo', default='data/geo/location_data.csv')
    p.add_argument('--n', default=500, type=int)
    args = p.parse_args()
    main(args.out, args.face_dir, args.labels, args.geo, args.n)
