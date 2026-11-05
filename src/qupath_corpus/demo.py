"""Generate a fully synthetic demo corpus for qupath-corpus-builder.

Creates fake "slide" images (textured background + roughly tubule-shaped
blobs) and matching QuPath-format JSON annotations (manually labelled PCT/DCT
polygons), then runs the builder end-to-end.  No real histology anywhere: this
exercises the exact pipeline shape users will run on their own data.

    python scripts/make_demo.py [--n-slides 3] [--out demo-data/output]

Outputs: a scratch tree with images/, jsons/, and the built corpus
(images/labels slide pairs) mirroring the builder's own output structure.
"""

import argparse
import json
import os

import cv2
import numpy as np

def _rng(r):
    return np.random.RandomState(r)


def make_fake_slide(r, height=200, width=260, n_tubules=8):
    """Textured 'H&E-ish' background plus distinct PCT/DCT blob clusters."""
    img = (r.randint(190, 235, (height, width, 3))
             .astype(np.uint8))
    # low-frequency texture so it doesn't look flat
    blur = cv2.GaussianBlur(
        r.randint(0, 40, (height, width, 3)).astype(np.uint8), (15, 15), 0)
    img = cv2.addWeighted(img, 0.9, blur, 0.1, 0)

    # nuclei-ish speckle
    speck = r.rand(height, width)
    img[speck > 0.985] = (60, 60, 140)

    polys = []
    for i in range(n_tubules):
        cy, cx = r.randint(30, height - 30), r.randint(30, width - 30)
        a, b = r.randint(6, 12), r.randint(5, 9)
        ang = r.uniform(0, np.pi)
        n = 14
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # jittered ellipse
        pts = np.stack([cx + a * np.cos(th) * (1 + 0.25 * r.randn(n)),
                        cy + b * np.sin(th + ang) * (1 + 0.25 * r.randn(n))],
                       axis=1).astype(np.float32)
        poly = pts.reshape(-1, 1, 2).astype(np.int32)
        color = (150, 90, 190) if i % 2 == 0 else (120, 140, 170)
        cv2.fillPoly(img, [poly], color)
        polys.append((pts, "PCT" if i % 2 == 0 else "DCT"))

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, polys


def write_qupath_json(path, polys):
    elements = []
    for pts, label in polys:
        elements.append({
            "closed": True, "type": "polyline",
            "points": [[float(x), float(y), 0.0] for x, y in pts],
            "label": {"value": label},
        })
    with open(path, "w") as f:
        json.dump({"annotation": {"elements": elements}}, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-slides", type=int, default=3)
    ap.add_argument("--out", default="demo-data/output")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = args.out
    img_dir = os.path.join(root, "raw_images")
    json_dir = os.path.join(root, "jsons")
    corpus = os.path.join(root, "corpus")
    for d in (img_dir, json_dir, corpus):
        os.makedirs(d, exist_ok=True)

    existing = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    if existing and not args.force:
        print(f"found {len(existing)} demo slides in {img_dir}; "
              f"use --force to regenerate")
    else:
        for s in range(args.n_slides):
            r = _rng(1000 + s)
            img, polys = make_fake_slide(r)
            cv2.imwrite(os.path.join(img_dir, f"{s + 1}_synthetic.png"), img)
            write_qupath_json(os.path.join(json_dir, f"tubules_{s + 1}.json"),
                              polys)
        print(f"wrote {args.n_slides} synthetic slide+JSON pairs to {root}")

    print("running builder...")
    from qupath_corpus.build_corpus import main as build_main
    code = build_main([
        "--json-dir", json_dir,
        "--image-dir", img_dir,
        "--out", corpus,
    ])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
