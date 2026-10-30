# Leakage report (P3-M00)

Images hashed: 1771 (train 1125, val 242, test 239, wild 164, user 1).
Near-duplicate groups (64/256-bit pHash over 8 flip/rotation variants, split rules, cross-crop): 1709, of which 54 have more than one image.

**Groups spanning more than one role (train / val / test / wild / user): 0**


Clean split clusters spanning splits (data/processed/split_meta.json): 0.

Augmentation is applied at load time to TRAIN rows only (LeakageError guards in leafdx.cnn.augment); no augmented copy is written to disk.
