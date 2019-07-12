# Semi-supervised learning on google command dataset
python3 experiments/paper/paper_mt_google.py

# Supervised learning on partial data google command dataset
python3 experiments/paper/paper_sup_google.py

# Semi-supervised learning on urban dataset
python3 experiments/paper/paper_mt_urban.py

# Supervised learning of partial data on urban command dataset
python3 experiments/paper/paper_sup_urban.py

# Semi-supervised learning on google command dataset with bg noise from google dataset
python3 experiments/paper/paper_mt_google_bg.py

# Semi-supervised learning on google command dataset with bg noise from urban dataset
python3 experiments/paper/paper_mt_google_bg_urban.py

# Semi-supervised learning on google command dataset with ict
python3 experiments/paper/paper_google_ict.py

# Supervised learning on google command dataset using ict only as augmentation
python3 experiments/paper/paper_sup_google_urban_aug.py