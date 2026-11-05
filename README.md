# qupath-corpus-builder

Turn QuPath polygon annotations + whole-slide histology images into aligned
**image/label training pairs** for segmentation-model fine-tuning.

Built in a medical-physics research pipeline (mini-pig renal-cortex tubule
PCT/DCT segmentation) to rescue 199 slides of hand-drawn annotations that were
sitting in QuPath instead of in a training set — but the annotation format is
generic QuPath, so any polygon label set works.

## What it does

For each slide:

1. Reads the QuPath JSON export (`{"annotation": {"elements": [...]}}`).
2. Rasterizes every labelled polygon into a label map (0 = background,
   1..K = your classes, in the order given to `--classes`).
3. Downsamples the raw image *and* the label map from native to working
   resolution (`--source-um-px` → `--target-um-px`, default 0.27 → 1.0 µm/px;
   images use INTER_AREA, labels use INTER_NEAREST so classes stay exact).
4. Writes aligned `slideNNN.png` (RGB) + label PNG pairs under `--out`.

Slides with no usable manual labels can be pseudo-labelled by the optional CNN
classifier (see below); otherwise they're skipped with a note.

## Install

```bash
pip install -e .    # numpy + opencv-headless + pillow only
pip install -e ".[svs]"         # + openslide, for .svs/.ndpi whole-slide files
pip install -e ".[classifier]"  # + torch/timm, for the CNN pseudo-labeler
```

## Quick start (synthetic demo, no real data)

The repo ships a **fully synthetic demo** — fake slides with fake PCT/DCT
polygons, zero real histology — so you can see the exact pipeline shape (and
verify a checkout works) before pointing it at your own data:

```bash
python scripts/make_demo.py --n-slides 3        # builds demo-data/output/
# or, after pip install -e .:
qupath-corpus-demo
```

This writes raw synthetic slides + QuPath JSONs and runs the builder end to
end, producing `images/slide001.png` + `labels/slide001.png` pairs.

## Real data

```bash
qupath-corpus \
  --json-dir  /path/to/qupath/jsons \
  --image-dir /path/to/slides \
  --out       /path/to/corpus \
  --classes PCT DCT
```

Requirements for your own run:
- `--json-dir`: QuPath exports named `*_<N>.json` (any prefix) whose
  `annotation.elements[]` carry `points` (≥4) and `label.value`.
- `--image-dir`: images named `<N>_*` matching each JSON's N. Plain
  `.png/.jpg/.tif` load via PIL; `.svs/.ndpi/...` need the `[svs]` extra.
- `--classes`: your label values, class index = position + 1. Default
  `PCT DCT` (so PCT→1, DCT→2) matches the renal-tubule use case.

## Optional CNN pseudo-labeler

If some slides have no manual labels, `--classifier model.pth` assigns classes
from a 4-channel (RGB+mask) `efficientnet_b0` trained to distinguish your
classes. Checkpoint format: a `timm.create_model('efficientnet_b0',
in_chans=4, num_classes=2)` state dict. Requires the `[classifier]` extra and
a real checkpoint — the repo ships no weights.

## Output format

```
out/
  images/slide001.png    # RGB at working resolution
  labels/slide001.png    # uint8: 0 bg, 1..K classes (exact after NEAREST resize)
```

## Notes on the tubule origin

Measured on annotated mini-pig kidney slides, pre-trained H&E models missed
~18% of tubules even at IoU 0.3, and recall fell 82% → 51% between IoU 0.3 and
0.7 — the annotation boundary style differed enough that only training on the
corpus fixed it. Slides 1–3 carried manual PCT/DCT labels; the rest were
pseudo-labelled by a classifier validated on a held-out slide (96% balanced
accuracy, 98% DCT recall), with hand-drawn polygons throughout. That strategy
is exactly what this tool automates for your data.

## License

MIT. Part of the author's PhD research (University of Florida Medical Physics).
