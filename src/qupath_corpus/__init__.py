"""qupath-corpus-builder: QuPath annotations -> image/label training corpus.

Reads QuPath polygon annotations (JSON) paired with whole-slide images and
writes aligned image/label PNG pairs at a working resolution, ready for
segmentation-model fine-tuning.  Supports manual labels and (optionally) a
CNN pseudo-labeler for unlabelled slides.

Built for a mini-pig renal-cortex tubule project, but the annotation format
is generic QuPath; any polygon label set works (see README).
"""

__version__ = "0.1.0"
