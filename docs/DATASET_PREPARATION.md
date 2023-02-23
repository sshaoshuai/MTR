# Waymo Dataset Preparation

**Step 1:** Download Waymo Open Motion Dataset (we use the `scenario protocol` form dataset), and organize the data as follows: 
```
MTR
├── data
│   ├── waymo
│   │   ├── scenario
│   │   │   ├──training
│   │   │   ├──validation
│   │   │   ├──testing
├── mtr
├── tools
```

**Step 2:** Install the [Waymo Open Dataset API](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md) as follows: 
```
pip install waymo-open-dataset-tf-2-6-0
```

**Step 3:** Preprocess the dataset:
```
cd mtr/datasets/waymo
python data_preprocess.py ../../../data/waymo/scenario/  ../../../data/waymo
```
These two paths indicate the raw data path and the output data path. 

Then, the processed data will be saved to `data/waymo/` directory as follows:
```
MTR
├── data
│   ├── waymo
│   │   ├── processed_scenarios_training
│   │   ├── processed_scenarios_validation
│   │   ├── processed_scenarios_training_infos.pkl
│   │   ├── processed_scenarios_val_infos.pkl
├── mtr
├── tools
```
