# Graph_encoding_on_DART

## Dataset
In this project, I am using [DART](https://github.com/Yale-LILY/dart). The processed file are in `data/processed_dart`.

## Environment


```
conda install -c pytorch pytorch=1.6.0
pip install torch-scatter==2.0.5 -f https://data.pyg.org/whl/torch-1.6.0+${CUDA}.html
pip install torch-sparse==0.6.8 -f https://data.pyg.org/whl/torch-1.6.0+${CUDA}.html
pip install torch-geometric==1.6.1
```

Finally, install the packages required:

```
pip install -r requirements.txt
```
`xml-python==0.4.3` is needed only for the preprocess of the raw DART data. You can always use the preprocessed data for the convinience.  

## Training

Train with different models:
 execute:
```
./train.sh <model> <data_dir> <gpu_id> 
```
 
Options for `<model>` are `t5-small`, `t5-base`, `t5-large`. 

Example:
```
./train.sh t5-base data/processed_dart 0
```


## Inference

For testing, run:
```
./test.sh <model> <checkpoint_folder> <data_dir> <gpu_id>
```

Example:
```
./test.sh t5-base ckpt-folder data/processed_dart 0
```


## Trained Model

Checkpoints can be found in `output` folder by diffrent experiments with prediction texts, metric.json and best models. 
