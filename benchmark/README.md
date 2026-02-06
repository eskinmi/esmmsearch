# Benchmark: Expedia Personalized Sort

Benchmarking the Two-Tower ESMM model on the [Expedia Personalized Sort (ICDM 2013)](https://www.kaggle.com/c/expedia-personalized-sort) Kaggle competition.

## Dataset Overview

| Metric         | Value            |
|----------------|------------------|
| Samples        | ~10M impressions |
| Sessions       | ~400K searches   |
| Click rate     | 4.49%            |
| Booking rate   | 2.78%            |
| **Evaluation** | **NDCG@38**      |
| Top scores     | 0.50 - 0.54      |

Relevance grades: **5** (booked), **1** (clicked), **0** (no interaction)

## Setup

### 1. Prerequisites

```bash
# Kaggle authentication (one of):
export KAGGLE_API_TOKEN=<your-api-key>
# OR: place kaggle.json in ~/.kaggle/

# Accept competition rules:
# https://www.kaggle.com/c/expedia-personalized-sort/rules
```

### 2. Download Dataset

```bash
make benchmark-setup
```

or for manual setup:

```bash
cd benchmark

# Download (~270MB compressed)
kaggle competitions download -c expedia-personalized-sort

# Extract (nested zips)
unzip expedia-personalized-sort.zip -d dataset/
cd dataset && unzip data.zip && rm -f *.zip && cd ..

# Verify
ls dataset/*.csv
# → dataset/train.csv  dataset/test.csv
```

### 4. Train

```bash
make benchmark-train
```

or manually:

```bash
# From project root
uv run esmmrank --data expedia --data-dir benchmark/dataset/processed \
    --epochs 20 --lr 0.0005 --batch-size 512
```

## Results

### Initial Iteration

| Metric  | Score    |
|---------|----------|
| NDCG@38 | `0.5270` |
| NDCG@10 | `0.4743` |
| PR-AUC  | `0.1635` |


## Feature Mapping

| Model Input         | Expedia Features                                                                |
|---------------------|---------------------------------------------------------------------------------|
| User categorical    | visitor_location_country_id, site_id                                            |
| User numerical      | hist_starrating, hist_adr, adults, children, rooms, stay_length, booking_window |
| Hotel categorical   | prop_id, prop_country_id, starrating, brand_bool, promotion_flag                |
| Hotel numerical     | review_score, location_score1/2, log_hist_price, price_usd, distance, affinity  |
| Context categorical | destination_id, saturday_night, random_bool                                     |
| Context numerical   | position (normalized)                                                           |

## References

- [Competition Page](https://www.kaggle.com/c/expedia-personalized-sort)
- [Winning Solutions Paper](https://arxiv.org/abs/1311.7679)
- [ESMM Paper](https://arxiv.org/abs/1804.07931)
