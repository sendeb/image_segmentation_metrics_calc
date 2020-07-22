### Directory/File Details

The following is a list of files and directories included in this repo:
- **metrics_utils**: The metrics_utils package is a custom package to calculate the following metrics to evaluate model performace:
	- Dice
	- IoU
	- Weighted Coverage
	- BoundF
- **calculate_metrics.py**: This was our evaluation script to validate the performance of our model on the Vaihingen and Bing datasets
- **pred_npys**: This directory contains an example of what an input to the calculate_metrics.py script looks like. Save predictions and ground truths as .npy arrays.

### Datasets:
We performed image segmentation on the following 2 datasets and used the metrics_utils package to validate our results.
- **Vaihingen**: Download the dataset [here](https://drive.google.com/drive/u/1/folders/1rWM9qL3PZjXkN80oy8rpHpHbqVzWObUb)
	 - **buildings_vaihingen_original**: Original dataset
	 - **buildings_vaihingen_multi_processed**: Use this version of the dataset to recreate our "multi instance segmentation" results
	 - **buildings_vaihingen_single_processed**: Use this version of the dataset to recreate our "single instance segmentation" results
- **Bing**: Download the dataset [here](https://drive.google.com/drive/u/1/folders/1H5zi1pISkrp6C4Lb_6ftzvu90NWvWY5f)
	- **buildings_bing_original**: Original dataset
	- **buildings_bing_multi_processed**: Use this version of the dataset to recreate our "multi instance segmentation" results
	- **buildings_bing_single_processed**: Use this version of the dataset to recreate our "single instance segmentation" results

### Metrics
The metrics_utils package is a custom package to calculate the following metrics to evaluate model performance:
- Dice
- IoU
- Weighted Coverage
- BoundF

Commands to run calculate_metrics.py:
- for Vaihingen:
```
python calculate_metrics.py --path /path/to/your/vaihingen/pred_npys --dataset V
```
- for Bing:
```
python calculate_metrics.py --path /path/to/your/bing/pred_npys --dataset B
```

