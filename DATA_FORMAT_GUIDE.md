# Data Format Guide

This document describes the expected data format for `data/face/` and `data/geo/` directories.

---

## Face Data Format (`data/face/`)

### Directory Structure

```
data/face/
├── images/                    # Face image files
│   ├── person_1_img1.jpg
│   ├── person_1_img2.jpg
│   ├── person_2_img1.jpg
│   ├── person_2_img2.jpg
│   └── ...
└── labels.csv                # Identity labels file
```

### Image Requirements

**Format:** JPEG, PNG, or other standard image formats
**Resolution:** Any (will be resized to 112×112 automatically)
**Quality:** Natural face images, frontal preferred for ArcFace
**Content:** Clear face images with minimal occlusion

### Label File Format

File: `data/face/labels.csv`

```csv
image_path,identity_id
images/person_1_img1.jpg,1
images/person_1_img2.jpg,1
images/person_2_img1.jpg,2
images/person_2_img2.jpg,2
images/person_3_img1.jpg,3
```

**Columns:**
- `image_path`: Relative path to image file (from `data/face/` directory)
- `identity_id`: Integer identity label (0, 1, 2, ..., N-1)

**Requirements:**
- CSV format with header row
- At least 10 images per identity for meaningful training
- Identity labels must be sequential integers starting from 0
- Paths relative to `data/face/` directory

### Example Dataset Creation

```python
import os
import pandas as pd
from PIL import Image
import numpy as np

# Create directory structure
os.makedirs('data/face/images', exist_ok=True)

# Create synthetic face images (for testing)
num_people = 5
images_per_person = 20

labels = []
for person_id in range(num_people):
    for img_idx in range(images_per_person):
        # Create random image (in practice, use real face images)
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save image
        img_path = f'data/face/images/person_{person_id}_img{img_idx}.jpg'
        img.save(img_path)
        
        # Record label
        labels.append({
            'image_path': f'images/person_{person_id}_img{img_idx}.jpg',
            'identity_id': person_id
        })

# Save labels CSV
df = pd.DataFrame(labels)
df.to_csv('data/face/labels.csv', index=False)
print(f"Created {len(labels)} face images for {num_people} identities")
```

### How the Code Loads Face Data

```python
from src.datasets import FaceDataset, build_face_dataloader

# Load dataset
dataset = FaceDataset(
    image_dir='data/face/images',
    label_file='data/face/labels.csv',
    image_size=112,
    split='train',
    add_degradation=True,  # Optional: add synthetic degradations
)

# Create dataloader
dataloader = build_face_dataloader(
    dataset=dataset,
    batch_size=128,
    shuffle=True,
)

# Iterate over batches
for batch_idx, (images, identities, quality_scores, paths) in enumerate(dataloader):
    # images: (batch_size, 3, 112, 112)
    # identities: (batch_size,) - integer labels
    # quality_scores: (batch_size,) - quality [0, 1]
    # paths: tuple of image paths
    pass
```

---

## 📍 Geofence Data Format (`data/geo/`)

### Directory Structure

```
data/geo/
└── geo_data.csv              # Location-based authentication data
```

### CSV File Format

File: `data/geo/geo_data.csv`

```csv
lat,lon,time_of_day,gps_accuracy,speed,label
40.7128,-74.0060,8,12.5,5.2,0
40.7128,-74.0059,9,10.3,3.1,0
40.7580,-73.9855,17,15.2,8.5,0
40.6892,-73.9195,4,25.1,35.2,1
40.5,-74.2,2,40.5,45.0,1
```

**Columns:**

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `lat` | float | -90 to 90 | Latitude (degrees) |
| `lon` | float | -180 to 180 | Longitude (degrees) |
| `time_of_day` | int | 0-23 | Hour of day (24-hour format) |
| `gps_accuracy` | float | 0+ | GPS accuracy estimate (meters) |
| `speed` | float | 0+ | Movement speed (m/s) |
| `label` | int | 0 or 1 | 0=legitimate, 1=attack |

### Example Dataset Creation

```python
import pandas as pd
import numpy as np

# Create synthetic geofence dataset
np.random.seed(42)

data = []

# Legitimate samples (inside geofences, normal times)
num_legit = 1000
for _ in range(num_legit):
    # Home location: (40.7128, -74.0060) with ±0.001 noise
    lat = np.random.normal(40.7128, 0.0005)
    lon = np.random.normal(-74.0060, 0.0005)
    
    # Normal access times (6-10 AM or 5-11 PM)
    if np.random.rand() > 0.5:
        time_of_day = np.random.randint(6, 11)  # Morning
    else:
        time_of_day = np.random.randint(17, 23)  # Evening
    
    gps_accuracy = np.random.uniform(5, 20)
    speed = np.random.uniform(0, 15)
    
    data.append({
        'lat': lat,
        'lon': lon,
        'time_of_day': time_of_day,
        'gps_accuracy': gps_accuracy,
        'speed': speed,
        'label': 0,  # Legitimate
    })

# Attack samples (outside geofences, unusual times, high speed)
num_attack = 500
for _ in range(num_attack):
    # Random location far away
    lat = np.random.uniform(40.5, 41.0)
    lon = np.random.uniform(-74.3, -73.7)
    
    # Unusual time (1-5 AM)
    time_of_day = np.random.randint(1, 6)
    
    gps_accuracy = np.random.uniform(20, 50)
    speed = np.random.uniform(25, 50)  # Suspicious high speed
    
    data.append({
        'lat': lat,
        'lon': lon,
        'time_of_day': time_of_day,
        'gps_accuracy': gps_accuracy,
        'speed': speed,
        'label': 1,  # Attack
    })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('data/geo/geo_data.csv', index=False)
print(f"Created {len(data)} geofence samples ({num_legit} legit, {num_attack} attacks)")
```

### How the Code Loads Geo Data

```python
from src.datasets import GeoDataset, build_geo_dataloader

# Load dataset with feature engineering
dataset = GeoDataset(
    csv_path='data/geo/geo_data.csv',
    split='train',
    geofences=[
        {'name': 'home', 'lat': 40.7128, 'lon': -74.0060, 'radius': 50.0},
        {'name': 'office', 'lat': 40.7580, 'lon': -73.9855, 'radius': 80.0},
    ],
    home_lat=40.7128,
    home_lon=-74.0060,
)

# Create dataloader
dataloader = build_geo_dataloader(
    dataset=dataset,
    batch_size=64,
    shuffle=True,
)

# Iterate over batches
for batch_idx, (features, labels) in enumerate(dataloader):
    # features: (batch_size, 9) - engineered features
    # labels: (batch_size,) - 0=legit, 1=attack
    # Features: [lat, lon, dist_to_home, dist_to_geofence, 
    #            time_sin, time_cos, gps_accuracy, speed, is_boundary]
    pass
```

### Feature Engineering Details

The GeoDataset automatically engineers 9 features from the raw data:

```python
features = [
    latitude,                    # Normalized GPS latitude
    longitude,                   # Normalized GPS longitude
    distance_to_home,           # Haversine distance to home (meters)
    distance_to_geofence,       # Min distance to any geofence (meters)
    time_of_day_sin,            # sin(2π * hour / 24)
    time_of_day_cos,            # cos(2π * hour / 24)
    gps_accuracy,               # GPS accuracy estimate (meters)
    speed,                       # Movement speed (m/s)
    is_boundary,                # 1 if near geofence boundary, 0 otherwise
]
```

All features are normalized to [0, 1] range for stable training.

---

## 🔄 Automatic Data Generation

If you don't have real data, use the built-in data generators:

### Generate Synthetic Face Data

```bash
python main.py --config config/face_baseline.yaml --mode train_face
```

This will:
1. Check if `data/face/labels.csv` exists
2. If not, generate synthetic face images in memory
3. Train the face verification model

### Generate Synthetic Geofence Data

```bash
python main.py --config config/geo_baseline.yaml --mode gen_geo_data
```

This generates `data/geo/geo_data.csv` with:
- 1000 legitimate samples (inside geofences, normal times)
- 500 attack samples (outside, boundary, unusual times, high speeds)

### Generate System Events

```bash
python main.py --config config/fusion_full_system.yaml --mode gen_system_events
```

This generates system-level events combining face + geo verification.

---

## 📊 Data Statistics Summary

### Face Data
- **Minimum:** 2 identities × 2 images = 4 samples
- **Recommended:** 50-100 identities × 20-50 images = 1000-5000 samples
- **Large-scale:** 1000+ identities × 50+ images = 50,000+ samples

### Geofence Data
- **Minimum:** 100 samples total
- **Recommended:** 1000 legitimate + 500 attacks = 1500 samples
- **Large-scale:** 10,000+ samples for statistical significance

---

## ⚙️ Using Real Data

### Replace Face Images

1. Place face images in `data/face/images/`
2. Create `data/face/labels.csv` with identity labels
3. Run training:
   ```bash
   python main.py --config config/face_baseline.yaml --mode train_face
   ```

### Replace Geofence Data

1. Prepare GPS logs or location authentication data
2. Format as CSV with columns: `lat, lon, time_of_day, gps_accuracy, speed, label`
3. Save to `data/geo/geo_data.csv`
4. Run training:
   ```bash
   python main.py --config config/geo_baseline.yaml --mode train_geo
   ```

---

## 🔍 Data Validation

Check your data format:

```python
import pandas as pd

# Check face labels
face_labels = pd.read_csv('data/face/labels.csv')
print(f"Face samples: {len(face_labels)}")
print(f"Identities: {face_labels['identity_id'].max() + 1}")
print(face_labels.head())

# Check geo data
geo_data = pd.read_csv('data/geo/geo_data.csv')
print(f"Geo samples: {len(geo_data)}")
print(f"Legitimate: {(geo_data['label'] == 0).sum()}")
print(f"Attacks: {(geo_data['label'] == 1).sum()}")
print(geo_data.describe())
```

---

## 🚀 Quick Start with Sample Data

Run the data generation scripts:

```bash
# Generate geofence data
python main.py --config config/geo_baseline.yaml --mode gen_geo_data

# View generated data
head -5 data/geo/geo_data.csv
```

This creates realistic synthetic data you can use to test the entire pipeline!
