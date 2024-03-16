# Norm-Aware Embedding for Efficient Person Search

## Results
|   Backbone   |  dataset | mAP | Top-1| config |
|--------------|---------|-----|------|--------|
|   R-50       | PRW | 44.3| 80.8 | [config](prw_nae.py)|

## train
    python tools/train.py configs/nae/prw_nae.py #PRW dataset