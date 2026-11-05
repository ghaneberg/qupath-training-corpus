"""Build an image/label segmentation training corpus from QuPath annotations.

Pipeline (per slide):
  1. Read the QuPath JSON annotations for the slide (polygon elements).
  2. Render each labelled polygon into a label map (0 = background, 1..K =
     classes in the order given by --classes).
  3. Downsample the raw slide image and the label map to the working
     resolution (--source-um-px / --target-um-px, default 0.27 -> 1.0 um/px).
  4. Write aligned ``slideNNN.png`` (RGB) + label PNG pairs to --out.

Annotation format (generic QuPath export):
    {"annotation": {"elements": [
        {"points": [[x0,y0,0], ...], "label": {"value": "PCT"},
         "closed": true, "type": "polyline"}, ...
    ]}}

Slides whose annotations carry no usable class labels are pseudo-labelled by
a CNN classifier when --classifier is given (efficientnet_b0, 4-channel
RGB+mask, num_classes=2, the author's lab format); otherwise they are skipped.
The classifier is an OPTIONAL extra: base install needs only numpy/cv2/PIL.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image

DEFAULT_CLASSES = ["PCT", "DCT"]   # class index = position + 1 (0 = background)
CLS_PATCH = 224                    # classifier input size
CLS_PAD = 32


def load_classifier(path: str, device: str = "cpu"):
    """Load the optional CNN pseudo-labeler (efficientnet_b0, 4-ch RGB+mask).

    Requires the ``classifier`` extra: pip install qupath-corpus[classifier]
    (torch + timm).  The checkpoint must be a state dict for
    timm.create_model('efficientnet_b0', in_chans=4, num_classes=2).
    """
    import torch
    import timm
    m = timm.create_model("efficientnet_b0", pretrained=False, in_chans=4,
                          num_classes=2, drop_rate=0.4)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval().to(device)


def classify(model, img, polys, batch=64):
    """PCT/DCT class (0/1) for each polygon, from RGB+mask patches."""
    import torch
    dev = next(model.parameters()).device
    out, buf, idx = [None] * len(polys), [], []

    def flush():
        if not buf:
            return
        t = torch.from_numpy(np.stack(buf).transpose(0, 3, 1, 2) / 255).float().to(dev)
        with torch.no_grad():
            p = model(t).argmax(1).cpu().numpy()
        for j, k in enumerate(idx):
            out[k] = int(p[j])
        buf.clear(); idx.clear()

    H, W = img.shape[:2]
    for i, xy in enumerate(polys):
        x0 = max(0, int(xy[:, 0].min()) - CLS_PAD)
        y0 = max(0, int(xy[:, 1].min()) - CLS_PAD)
        x1 = min(W, int(xy[:, 0].max()) + CLS_PAD)
        y1 = min(H, int(xy[:, 1].max()) + CLS_PAD)
        if x1 - x0 < 4 or y1 - y0 < 4:
            out[i] = 0
            continue
        rgb = img[y0:y1, x0:x1]
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        sh = xy.copy(); sh[:, 0] -= x0; sh[:, 1] -= y0
        cv2.fillPoly(m, [sh], 255)
        patch = np.concatenate([rgb, m[:, :, None]], axis=2)
        buf.append(cv2.resize(patch, (CLS_PATCH, CLS_PATCH),
                              interpolation=cv2.INTER_LINEAR))
        idx.append(i)
        if len(buf) >= batch:
            flush()
    flush()
    return [0 if v is None else v for v in out]


def load_slide(path: str) -> np.ndarray:
    """Read a whole-slide image as an RGB uint8 array (level 0).

    Supports plain images (.png/.jpg/.tif via PIL) and whole-slide formats
    (.svs/.ndpi/.qptiff via openslide, requires the ``svs`` extra).
    """
    if path.lower().endswith((".svs", ".ndpi", ".qptiff", ".mrxs", ".tiff")):
        try:
            import openslide
        except ImportError:
            sys.exit("openslide required for whole-slide files: "
                     "pip install qupath-corpus[svs]")
        s = openslide.OpenSlide(path)
        arr = np.array(s.read_region((0, 0), 0, s.dimensions).convert("RGB"))
        s.close()
        return arr
    return np.asarray(Image.open(path).convert("RGB"))


def parse_annotations(json_path):
    """Return (polys, labels) from a QuPath JSON file.

    polys: list of (N,2) int32 point arrays; labels: list of str (raw value).
    """
    with open(json_path) as f:
        els = json.load(f)["annotation"]["elements"]
    polys, labels = [], []
    for e in els:
        pts = e.get("points", [])
        if len(pts) < 4 or e.get("type", "polyline") != "polyline":
            continue
        polys.append(np.array([[p[0], p[1]] for p in pts], np.int32))
        labels.append((e.get("label") or {}).get("value", "").upper())
    return polys, labels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", required=True, help="dir of tubules_NNN.json QuPath exports")
    ap.add_argument("--image-dir", required=True, help="dir of raw slide images (*.svs/png/...)")
    ap.add_argument("--out", default="output", help="output corpus dir")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="class labels, class index = position+1 (default: PCT DCT)")
    ap.add_argument("--source-um-px", type=float, default=0.27,
                    help="native resolution of source images (um/px)")
    ap.add_argument("--target-um-px", type=float, default=1.0,
                    help="working resolution (um/px) of the corpus")
    ap.add_argument("--classifier", default=None,
                    help="optional CNN checkpoint for pseudo-labelling unlabelled slides")
    ap.add_argument("--force", action="store_true", help="overwrite existing slide pairs")
    args = ap.parse_args(argv)

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)

    clf = load_classifier(args.classifier) if args.classifier else None

    json_paths = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
    slides = []
    for p in json_paths:
        m = re.search(r"(\d+)\.json$", os.path.basename(p))
        if m:
            slides.append((int(m.group(1)), p))
    if not slides:
        sys.exit(f"no tubules_NNN.json found under {args.json_dir}")

    scale = args.source_um_px / args.target_um_px
    class_index = {name: i + 1 for i, name in enumerate(args.classes)}

    n_pct = n_dct = n_slide = 0
    for n, jp in slides:
        ip = os.path.join(args.out, "images", f"slide{n:03d}.png")
        lp = os.path.join(args.out, "labels", f"slide{n:03d}.png")
        if os.path.exists(ip) and os.path.exists(lp) and not args.force:
            continue
        hits = [p for p in glob.glob(os.path.join(args.image_dir, "*"))
                if re.match(rf"^{n}_", os.path.basename(p))
                and not os.path.isdir(p)]
        if not hits:
            print(f"slide {n:3d}: no image, skipped")
            continue
        img = load_slide(hits[0])
        H, W = img.shape[:2]

        polys, manual = parse_annotations(jp)

        known = [m for m in manual if m in class_index]
        if len(known) == len(polys) and len(polys):
            cls = [class_index[m] for m in manual]
            src = "manual"
        elif clf is not None:
            cls = classify(clf, img, polys)          # 0/1 from the CNN
            cls = [class_index[args.classes[c]] for c in cls]
            src = "classifier"
        else:
            print(f"slide {n:3d}: no usable labels and no --classifier, skipped")
            continue

        lab = np.zeros((H, W), np.uint8)
        for xy, c in zip(polys, cls):
            cv2.fillPoly(lab, [xy], int(c))

        small = cv2.resize(img, (int(W * scale), int(H * scale)),
                           interpolation=cv2.INTER_AREA)
        labs = cv2.resize(lab, (int(W * scale), int(H * scale)),
                          interpolation=cv2.INTER_NEAREST)
        Image.fromarray(small).save(ip)
        Image.fromarray(labs).save(lp)

        n_ = sum(1 for c in cls if c == 1); d_ = sum(1 for c in cls if c == 2)
        n_pct += n_; n_dct += d_; n_slide += 1
        print(f"slide {n:3d}: {len(polys):4d} profiles  "
              f"{args.classes[0]} {n_:4d}  {args.classes[1]} {d_:4d}  [{src}]",
              flush=True)

    tot = n_pct + n_dct
    print(f"\n{n_slide} slides -> {args.out}")
    if tot:
        print(f"{tot} labelled profiles: {args.classes[0]} {n_pct} "
              f"({100*n_pct/tot:.1f}%)  {args.classes[1]} {n_dct} "
              f"({100*n_dct/tot:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
