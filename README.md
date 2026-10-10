# Detect-Species — camera-trap species classification

Counts which animal species appear in a month of unlabelled camera-trap footage (May 2022), focusing on **gazelle** (prey) and **jackal** (predator).

All the code is in [`pipeline.py`](pipeline.py) and reads top to bottom in five stages.

```
labelled crops ──► 1. clean ──► 2. split 70/15/15 ──► 3. train + tune YOLO classifier ──► 4. test once
                                                                  │
field frames ──► 5. MegaDetector ──► animal crops ──► classify ──► reject low-confidence ──► species counts
```

## Data

- **Training set:** 2,500 labelled animal crops in 11 classes. After cleaning, 2,137 images remain in 8 classes: cow, birds, wild boar, jackal, gazelle, porcupine, fallow deer and fox. They are split into 1,496 train, 321 validation and 320 test images.
- **Field set:** 2,423 unlabelled full frames from May 2022.

## Method

| Stage | What is done | Why |
|---|---|---|
| Cleaning | Drop `undefined`, `hyena` (1 image), `wolf` (9) and 9 crops that are black even after auto-contrast | Not a species, or too rare to evaluate. `fox` (47) is kept so field foxes aren't forced into `jackal`. |
| Split | 70 / 15 / 15 per class. Near-duplicate burst frames (8×8 average hash, ≤ 5 bits apart) are kept in the same split | Prevents a burst frame in test from having a near-copy in train |
| Model | YOLO26n-cls pretrained on ImageNet, fine-tuned for 30 epochs at 224 px with Ultralytics' default augmentation | |
| Tuning | `batch` 8/16/32, `imgsz` 160/224/320 and `weight_decay` 0/0.0005/0.005, changed one at a time | Selection uses validation only, scored by mean(macro precision, macro recall, top-5) because classes are imbalanced |
| Field detection | MegaDetector v5a, keeping "animal" boxes with confidence ≥ 0.2 | The classifier expects single-animal crops |
| Rejection | A crop is marked `uncertain` when its top-1 confidence is below tau = 0.82 (5th percentile of confidence among correct validation predictions) | A softmax classifier always picks a class, even for an empty box or an animal it was never trained on |

## Results

**Model selection (validation):** `batch_8` scored highest (macro P 0.906, macro R 0.892, top-1 0.950), mainly because it did better on the rare classes. Changing `imgsz` and `weight_decay` didn't help.

**Test set (320 images, scored once):**

| model | macro precision | macro recall | top-1 |
|---|---|---|---|
| baseline (batch 16) | 0.940 | 0.946 | 0.969 |
| selected (batch 8) | 0.938 | 0.937 | 0.966 |

On the test set, the two models differ by about two images. With only 7–9 test images in each rare class, one image moves that class's recall by 11–14 points.

**Target species on test (selected model):**

| true \ predicted | gazelle | fallow deer | jackal | other |
|---|---|---|---|---|
| gazelle (22) | **19** | 1 | 0 | 2 |
| fallow deer (9) | 1 | **7** | 0 | 1 |
| jackal (33) | 0 | 0 | **29** | 4 |

Gazelle and jackal are never confused with each other. The only real confusion is gazelle ↔ fallow deer, which look alike. No other species was ever predicted as jackal (precision 1.00).

**Field data:** 2,423 frames produced 993 animal crops (from 710 frames), and 204 of them (20.5%) were rejected as uncertain. Of the 789 crops kept, 685 are wild boar (457) or birds (228). **Jackal (18 crops) outnumbers gazelle (13)**. The field crops are less clean than the training crops (partial, distant or empty boxes), so these counts show relative presence, not the number of individuals.

## Running

```bash
pip install -r requirements.txt
python pipeline.py      # expects data/ as described at the top of pipeline.py; writes to work/
```
