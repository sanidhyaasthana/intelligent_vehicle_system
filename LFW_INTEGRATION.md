# LFW Dataset Integration Summary

## What You Need to Do

Your project now has full support for the LFW (Labeled Faces in the Wild) dataset!

### Step 1: Prepare the Dataset (One Command)

```bash
python setup_lfw.py
```

This will:
- ✓ Scan the `data/face/lfw-deepfunneled/` directory
- ✓ Validate all images
- ✓ Create `data/face/labels.csv` with identity mappings
- ✓ Print dataset statistics

### Step 2: Train Face Models

```bash
# Baseline (fixed margin)
python main.py --config config/face_baseline.yaml --mode train_face

# Or with adaptive margin
python main.py --config config/face_adaptive.yaml --mode train_face
```

### Step 3: Evaluate

```bash
python main.py --config config/face_baseline.yaml --mode eval_face
```

---

## New Files Created

### 1. **src/utils/lfw_utils.py** (560+ lines)
Complete LFW handling utilities:
- `LFWDatasetHandler` - Scans, validates, and processes LFW
- `prepare_lfw_dataset()` - One-line dataset preparation
- `create_subset()` - Create smaller subsets for testing
- `get_identity_distribution()` - Dataset statistics

### 2. **scripts/prepare_lfw.py** (95 lines)
CLI tool for LFW preparation with options:
```bash
python scripts/prepare_lfw.py                                    # Full dataset
python scripts/prepare_lfw.py --subset 100 2000                 # 100 people, 2000 images
python scripts/prepare_lfw.py --min-images 5                    # Min 5 images/person
```

### 3. **setup_lfw.py** (100 lines)
Quick one-command setup script (recommended):
```bash
python setup_lfw.py
```

### 4. **LFW_SETUP_GUIDE.md** (300+ lines)
Documentation includes:
- Quick start instructions
- Configuration examples
- Troubleshooting guide
- Expected results
- Python API examples

---

## Key Features

### Automatic Dataset Handling
```python
from src.utils.lfw_utils import prepare_lfw_dataset

df = prepare_lfw_dataset('data/face/lfw-deepfunneled')
# Returns DataFrame with 13,233 images from 5,749 people
```

### Image Validation
- Checks if images are readable (corrupted file detection)
- Validates minimum size (configurable)
- Removes invalid samples automatically

### Flexible Subsets
```python
from src.utils.lfw_utils import create_subset

# Create subset for faster testing
subset = create_subset(df, num_identities=100, num_samples=2000)
subset.to_csv('data/face/labels_subset.csv', index=False)
```

### Statistics & Diagnostics
```python
handler = LFWDatasetHandler('data/face/lfw-deepfunneled')
identity_images = handler.scan_dataset()
valid_images = handler.validate_images(identity_images)
handler.print_statistics(valid_images)
```

---

## Data Format (Automatic)

After running `python setup_lfw.py`, you'll get:

**data/face/labels.csv:**
```csv
image_path,identity_id
lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg,0
lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0002.jpg,0
lfw-deepfunneled/Abigail_Breslin/Abigail_Breslin_0001.jpg,1
...
```

This is automatically created from the directory structure.

---

## Expected Results

**Dataset Statistics:**
- People: 5,749
- Images: 13,233
- Images/person: 2.31 (avg)
- Size: ~200 MB

**Training Performance:**
- Training time: 2-4 hours (with GPU)
- EER on LFW test set: ~2-5% (baseline)
- EER on LFW test set: ~1-3% (adaptive margin)
- AUC: 98-99%

---

## Troubleshooting

### "lfw-deepfunneled not found"
Make sure the dataset is at:
```
data/face/lfw-deepfunneled/
```

Check with:
```bash
ls data/face/lfw-deepfunneled/ | head -10
```

### "Out of memory"
Reduce batch size in config or create a subset:
```bash
python setup_lfw.py --subset 100 1000  # Creates smaller version
```

### "Very slow data loading"
Increase num_workers in config:
```yaml
training:
  num_workers: 8  # Increase from 4
```

---

## Usage Examples

### Complete Training Pipeline
```bash
# 1. Setup dataset
python setup_lfw.py

# 2. Train with baseline margin
python main.py --config config/face_baseline.yaml --mode train_face

# 3. Train with adaptive margin
python main.py --config config/face_adaptive.yaml --mode train_face

# 4. Evaluate both models
python main.py --config config/face_baseline.yaml --mode eval_face

# 5. View results
cat results/face/face_baseline/metrics.csv
```

### Using Python API
```python
from src.datasets import FaceDataset, build_face_dataloader
from src.utils.lfw_utils import prepare_lfw_dataset

# Step 1: Prepare LFW
prepare_lfw_dataset('data/face/lfw-deepfunneled')

# Step 2: Load dataset
dataset = FaceDataset(
    image_dir='data/face/lfw-deepfunneled',
    label_file='data/face/labels.csv',
    split='train',
)

# Step 3: Create dataloader
dataloader = build_face_dataloader(dataset, batch_size=128)

# Step 4: Iterate
for images, identities, quality_scores, paths in dataloader:
    print(f"Batch: {images.shape}")  # (128, 3, 112, 112)
```

### Creating Custom Subsets
```python
from src.utils.lfw_utils import create_subset
import pandas as pd

df = pd.read_csv('data/face/labels.csv')

# Create multiple subsets for different experiments
subsets = [
    ('labels_100.csv', 100, None),      # 100 identities, all images
    ('labels_500.csv', None, 500),      # 500 images, all identities
    ('labels_50_500.csv', 50, 500),     # 50 identities, 500 images
]

for output, num_ids, num_samples in subsets:
    create_subset(df, num_identities=num_ids, 
                  num_samples=num_samples, 
                  output_path=f'data/face/{output}')
```

---

## Command Reference

| Command | Purpose |
|---------|---------|
| `python setup_lfw.py` | Prepare LFW dataset (creates labels.csv) |
| `python scripts/prepare_lfw.py` | Detailed preparation script with options |
| `python scripts/prepare_lfw.py --subset 100 2000` | Create 100-person subset |
| `python main.py --config config/face_baseline.yaml --mode train_face` | Train face model |
| `python main.py --config config/face_adaptive.yaml --mode train_face` | Train with adaptive margin |
| `python main.py --config config/face_baseline.yaml --mode eval_face` | Evaluate face model |
| `bash scripts/run_face_baseline.sh` | Full pipeline (train + eval) |
| `bash scripts/run_face_adaptive.sh` | Adaptive pipeline (train + eval) |

---

## Next Steps

1. **Prepare dataset:**
   ```bash
   python setup_lfw.py
   ```

2. **Start training:**
   ```bash
   python main.py --config config/face_baseline.yaml --mode train_face
   ```

3. **Monitor progress:**
   ```bash
   # View metrics while training
   watch "tail results/face/face_baseline/metrics.csv"
   ```

4. **Evaluate results:**
   ```bash
   python main.py --config config/face_baseline.yaml --mode eval_face
   cat results/face/face_baseline/summary.csv
   ```

For detailed documentation, see:
- `LFW_SETUP_GUIDE.md` - Setup guide with detailed instructions
- `DATA_FORMAT_GUIDE.md` - Data format specifications
- `IMPLEMENTATION_GUIDE.md` - Technical implementation details
