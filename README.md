# Detect-Species

## Background

We were asked to train and validate an image model that identifies animal species in camera-trap photos, then use it to count which species appear in a month of new, unlabelled field footage (May 2022). The key question: do jackals (predators) outnumber gazelles (prey)?

![one example per species](assets/species.jpg)

## How it works

```mermaid
flowchart LR
    A[📷 2,500 labelled crops] --> B[Clean<br/>8 species kept]
    B --> C[Split 70/15/15<br/>bursts kept together]
    C --> D[Fine-tune YOLO<br/>+ tune 3 settings]
    D --> E[Test once<br/>top-1 96.6%]
    F[🌲 2,423 field frames] --> G[MegaDetector<br/>finds animals]
    G --> H[993 crops]
    E -. best model .-> I
    H --> I[Classify]
    I --> J{Confident?}
    J -- yes --> K[Species count]
    J -- no --> L[uncertain]
```

## Three problems the pipeline handles

**1. Burst frames.** Camera traps fire several near-identical shots. If one lands in train and its twin in test, the test score is inflated. Near-duplicates are therefore kept together in one split.

![two burst frames](assets/burst.jpg)

**2. Look-alikes.** Gazelle (top row) and fallow deer (middle row) mostly differ by the deer's spots and antlers. Jackal (bottom row) is the predator they're compared with.

![gazelle, fallow deer, jackal](assets/lookalikes.jpg)

**3. Animals the model was never trained on.** A classifier always picks one of its species, even for an empty box or a hyena. Crops scored below 82% confidence are set aside as `uncertain`.

## Results

**Test set:** 309 of 320 images correct, and gazelle is never confused with jackal.

<img src="assets/confusion.png" width="480" alt="confusion matrix on the test set">

**May 2022 field data:** of 993 animal crops, jackal outnumbers gazelle.

```mermaid
xychart-beta
    x-axis ["jackal", "gazelle"]
    y-axis "confident crops" 0 --> 20
    bar [18, 13]
```

<sub>The rest are wild boar (457), birds (228), other species (73) and uncertain (204).</sub>

---
Code: [`pipeline.py`](pipeline.py), five steps that read top to bottom. Run it with `pip install -r requirements.txt && python pipeline.py`.
