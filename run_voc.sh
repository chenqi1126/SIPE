
# Step 1. Train SIPE for localization maps.

# 1.1 train sipe
python train_resnet50_SIPE.py
# 1.2 obtain localization maps
python make_cam.py
# 1.3 evaluate localization maps
python eval_cam.py


# Step 2. Train IRN for pseudo labels.

# 2.1 generate ir label
python cam2ir.py
# 2.2 train irn
python train_irn.py
# 2.3 make pseudo labels
python make_seg_labels.py