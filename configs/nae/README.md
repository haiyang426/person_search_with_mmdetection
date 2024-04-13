# Norm-Aware Embedding for Efficient Person Search

## Results
|   Backbone   |  dataset | mAP | Top-1| config |
|--------------|---------|-----|------|--------|
|   R-50       | PRW | 45.4| 81.6 | [config](prw_nae.py)|

## train
    python tools/train.py configs/nae/prw_nae.py #PRW dataset