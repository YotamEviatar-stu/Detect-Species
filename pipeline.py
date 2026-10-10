"""
Camera-trap species classification
==================================

Goal: count which animal species appear in a month of unlabelled camera-trap footage
(May 2022), with particular attention to gazelle (prey) and jackal (predator).

The pipeline has five stages:

    1. Clean the labelled training crops.
    2. Split them into train / validation / test without leaking burst frames.
    3. Fine-tune a YOLO image classifier and tune three hyperparameters.
    4. Evaluate the chosen model once on the held-out test set.
    5. Run it on the field footage: detect animals, classify each crop,
       and set aside predictions the model is not confident about.

Input layout (DATA_DIR):
    training_dataset/              labelled animal crops (*.JPG)
    training_dataset_labels.csv    columns: image_name, label
    new_dataset/                   unlabelled full field frames (*.JPG)
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

DATA_DIR = Path("data")
WORK_DIR = Path("work")

IMG_DIR    = DATA_DIR / "training_dataset"
LABEL_CSV  = DATA_DIR / "training_dataset_labels.csv"
FIELD_DIR  = DATA_DIR / "new_dataset"
SPLIT_DIR  = WORK_DIR / "dataset"        # train/val/test/<class>/ folders, the layout YOLO expects
RUNS_DIR   = WORK_DIR / "runs"
CROP_DIR   = WORK_DIR / "field_crops"
RESULTS    = WORK_DIR / "results"

SEED = 42


# =====================================================================================
# 1. Cleaning
# =====================================================================================
#
# Removed before anything else:
#   - 'undefined' is not a species.
#   - 'hyena' (1 image) and 'wolf' (9 images) are too rare to train on or to evaluate:
#     a 15% test share would give them 0 and 1 test images.
#   - Nine crops that stay completely black even after auto-contrast (found by visual
#     inspection of the darkest images).
# 'fox' (47 images) is kept despite being small: without a fox class, field foxes would be
# forced into the nearest class, most likely jackal, one of the two target species.

DROP_CLASSES = ["undefined", "hyena", "wolf"]
BLACK_IMAGE_IDS = ["00230861", "59137014", "45693490", "84977012",
                   "43260595", "14060955", "92776703", "92867346", "49790795"]

labels = pd.read_csv(LABEL_CSV)
image_id = labels["image_name"].str.extract(r"(\d+)\.JPG$", expand=False)
clean = labels[~labels["label"].isin(DROP_CLASSES) & ~image_id.isin(BLACK_IMAGE_IDS)].reset_index(drop=True)

CLASSES = clean["label"].value_counts().index.tolist()
print(f"{len(labels)} labelled images -> {len(clean)} after cleaning, {len(CLASSES)} classes")


# =====================================================================================
# 2. Train / validation / test split (70 / 15 / 15)
# =====================================================================================
#
# Camera traps fire in bursts, producing several near-identical frames of the same animal.
# If one burst frame lands in train and another in test, the test score is inflated,
# because the model has effectively already seen that test image.
#
# So near-duplicates are first grouped, and whole groups are assigned to a split:
#   - Each image is reduced to an 8x8 "average hash": 64 bits, each saying whether that
#     pixel is brighter than the image mean.
#   - Two images whose hashes differ in <= 5 bits, and which carry the same label, are
#     treated as one burst. Groups are formed transitively (union-find).
#   - Within each class, groups are dealt out largest-first to whichever split is furthest
#     below its 70/15/15 quota, so every class gets the right proportions.

def average_hash(path):
    gray = np.asarray(Image.open(path).convert("L").resize((8, 8)), dtype=np.float32)
    return (gray > gray.mean()).flatten()


def group_near_duplicates(df, max_bits=5):
    hashes = np.stack([average_hash(IMG_DIR / n) for n in df["image_name"]])
    labels = df["label"].to_numpy()
    parent = list(range(len(df)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(df)):
        distance = (hashes[i + 1:] != hashes[i]).sum(axis=1)
        for j in np.where(distance <= max_bits)[0] + i + 1:
            if labels[i] == labels[j]:          # cross-label matches are hash collisions, not bursts
                parent[root(i)] = root(j)
    return [root(i) for i in range(len(df))]


def split_quota(n, fractions={"train": 0.70, "val": 0.15, "test": 0.15}):
    """How many of n images each split should get (largest-remainder rounding, sums to n)."""
    raw = {s: n * f for s, f in fractions.items()}
    quota = {s: int(v) for s, v in raw.items()}
    for s in sorted(raw, key=lambda s: raw[s] - quota[s], reverse=True)[: n - sum(quota.values())]:
        quota[s] += 1
    return quota


clean["group"] = group_near_duplicates(clean)
clean["split"] = None

for cls in CLASSES:
    in_class = clean[clean["label"] == cls]
    group_sizes = (in_class.groupby("group").size()
                   .sample(frac=1, random_state=SEED)            # random tie-break between equal sizes
                   .sort_values(ascending=False, kind="stable"))
    quota = split_quota(len(in_class))
    filled = {s: 0 for s in quota}
    for group, size in group_sizes.items():
        target = max(quota, key=lambda s: (quota[s] - filled[s], quota[s]))
        clean.loc[clean["group"] == group, "split"] = target
        filled[target] += size

print(pd.crosstab(clean["label"], clean["split"])[["train", "val", "test"]])

# Copy the images into the folder layout YOLO reads: <split>/<class>/<image>.
# Images are copied unchanged; YOLO resizes, normalises and augments them in memory.
for row in clean.itertuples():
    dest = SPLIT_DIR / row.split / row.label
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(IMG_DIR / row.image_name, dest / row.image_name)


# =====================================================================================
# 3. Training and hyperparameter tuning
# =====================================================================================
#
# Model: YOLO26n-cls, pretrained on ImageNet, fine-tuned with all layers unfrozen.
# Baseline settings: 224 px input, batch 16, weight decay 0.0005, 30 epochs.
# Ultralytics' default augmentation is used (random resized crop, horizontal flip,
# RandAugment, random erasing), which suits camera-trap crops: animals are not centred,
# face either way, and appear under varying light.
#
# Tuning: each of three hyperparameters is moved one step down and one step up while
# everything else stays fixed, giving 6 runs plus the baseline.

BASE_SETTINGS = dict(data=str(SPLIT_DIR), project=str(RUNS_DIR), imgsz=224, batch=16,
                     weight_decay=0.0005, epochs=30, seed=SEED, deterministic=True)

RUNS = {
    "baseline":  {},
    "batch_8":   dict(batch=8),
    "batch_32":  dict(batch=32),
    "imgsz_160": dict(imgsz=160),
    "imgsz_320": dict(imgsz=320),
    "wd_0.0":    dict(weight_decay=0.0),
    "wd_0.005":  dict(weight_decay=0.005),
}

for name, change in RUNS.items():
    YOLO("yolo26n-cls.pt").train(**{**BASE_SETTINGS, **change}, name=name, exist_ok=True)


def evaluate(run_name, split):
    """Confusion matrix (rows = true class, columns = predicted) and per-class precision / recall."""
    model = YOLO(RUNS_DIR / run_name / "weights" / "best.pt")
    metrics = model.val(data=str(SPLIT_DIR), split=split, verbose=False)
    cm = metrics.confusion_matrix.matrix.T
    precision = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    names = [model.names[i] for i in range(len(model.names))]
    return dict(names=names, cm=cm, precision=precision, recall=recall,
                top1=metrics.top1, top5=metrics.top5)


# The winner is chosen on VALIDATION only. Classes are imbalanced (cow is 31% of images,
# fox about 2%), so plain accuracy would reward ignoring the rare classes; the score is
# therefore the mean of macro precision, macro recall and top-5 accuracy.

scores = {}
for name in RUNS:
    ev = evaluate(name, "val")
    scores[name] = dict(macro_precision=ev["precision"].mean(), macro_recall=ev["recall"].mean(),
                        top1=ev["top1"], top5=ev["top5"])
scores = pd.DataFrame(scores).T
scores["score"] = scores[["macro_precision", "macro_recall", "top5"]].mean(axis=1)
scores = scores.sort_values("score", ascending=False)
print(scores.round(3))

WINNER = scores.index[0]
print("selected model:", WINNER)


# =====================================================================================
# 4. Test-set evaluation (read once)
# =====================================================================================

RESULTS.mkdir(parents=True, exist_ok=True)
scores.round(4).to_csv(RESULTS / "validation_scores.csv")

test = evaluate(WINNER, "test")
names = test["names"]
pd.DataFrame(test["cm"].astype(int), index=names, columns=names).to_csv(RESULTS / "test_confusion_matrix.csv")
pd.DataFrame({"precision": test["precision"], "recall": test["recall"]}, index=names) \
  .round(3).to_csv(RESULTS / "test_per_class.csv")

gazelle, jackal = names.index("gazelle"), names.index("jackal")
print(f"test: macro precision {test['precision'].mean():.3f}, macro recall {test['recall'].mean():.3f}, "
      f"top-1 {test['top1']:.3f}")
print(f"gazelle <-> jackal confusions: {int(test['cm'][gazelle, jackal])} / {int(test['cm'][jackal, gazelle])}")


# =====================================================================================
# 5. Field data: detect, classify, reject
# =====================================================================================
#
# 5a. Detection. The field frames are full scenes, but the classifier was trained on
#     crops of single animals. MegaDetector v5a finds the animals; every 'animal' box
#     with confidence >= 0.2 is cropped out.

from megadetector.detection.run_detector_batch import load_and_run_detector_batch

DETECTION_THRESHOLD = 0.2

frames = sorted(str(f) for f in FIELD_DIR.glob("*.JPG"))
detections = load_and_run_detector_batch("MDV5A", frames, confidence_threshold=DETECTION_THRESHOLD)

CROP_DIR.mkdir(parents=True, exist_ok=True)
crops = []
for frame in detections:
    image = None
    for k, det in enumerate(frame.get("detections", [])):
        if det["category"] != "1" or det["conf"] < DETECTION_THRESHOLD:   # category 1 = animal
            continue
        if image is None:
            image = Image.open(frame["file"]).convert("RGB")
        W, H = image.size
        x, y, w, h = det["bbox"]                                          # relative coordinates
        crop_name = f"{Path(frame['file']).stem}_{k}.JPG"
        image.crop((int(x * W), int(y * H), int((x + w) * W), int((y + h) * H))).save(CROP_DIR / crop_name)
        crops.append(dict(crop=crop_name, frame=Path(frame["file"]).name, detection_conf=det["conf"]))
field = pd.DataFrame(crops)
print(f"{len(frames)} frames -> {len(field)} animal crops")

# 5b. Classification with the selected model.

classifier = YOLO(RUNS_DIR / WINNER / "weights" / "best.pt")


def class_probabilities(paths):
    results = classifier.predict([str(p) for p in paths], stream=True, verbose=False)
    return np.stack([r.probs.data.cpu().numpy() for r in results])


probs = class_probabilities(CROP_DIR / c for c in field["crop"])
field["species"] = [names[i] for i in probs.argmax(axis=1)]
field["confidence"] = probs.max(axis=1)

# 5c. Rejection. A softmax classifier always picks one of its classes, even for a crop
#     that shows no identifiable animal or an animal outside the label set (hyena, wolf).
#     Crops whose top-1 confidence is below a threshold tau are labelled 'uncertain'.
#
#     tau is set on validation: the 5th percentile of top-1 confidence among the
#     predictions that were correct, i.e. it keeps 95% of correct calls.

val_files = [(f, c) for c in names for f in sorted((SPLIT_DIR / "val" / c).glob("*"))]
val_probs = class_probabilities(f for f, _ in val_files)
val_correct = val_probs.argmax(axis=1) == np.array([names.index(c) for _, c in val_files])
TAU = float(np.percentile(val_probs.max(axis=1)[val_correct], 5))
print(f"rejection threshold tau = {TAU:.3f}")

# Sanity check of tau on images the model has never seen a class for: the dropped
# hyena / wolf / undefined images. A good threshold should reject most of them.
unseen = labels[labels["label"].isin(DROP_CLASSES)]
unseen_conf = class_probabilities(IMG_DIR / n for n in unseen["image_name"]).max(axis=1)
print(pd.Series(unseen_conf < TAU, index=unseen["label"].values).groupby(level=0).agg(["size", "sum"])
        .rename(columns={"size": "n", "sum": "rejected"}))

field["species_final"] = np.where(field["confidence"] < TAU, "uncertain", field["species"])

# 5d. Species counts. Each count is shown next to that class's test precision and recall,
#     so a reader can judge how far each number can be trusted.

counts = pd.DataFrame({
    "raw":       field["species"].value_counts().reindex(names).fillna(0).astype(int),
    "kept":      field["species_final"].value_counts().reindex(names).fillna(0).astype(int),
    "test_precision": pd.Series(test["precision"], index=names).round(2),
    "test_recall":    pd.Series(test["recall"], index=names).round(2),
})
print(counts)
print(f"{(field['species_final'] == 'uncertain').sum()} of {len(field)} crops rejected as uncertain")

field.to_csv(RESULTS / "field_predictions.csv", index=False)
counts.to_csv(RESULTS / "field_species_counts.csv")
