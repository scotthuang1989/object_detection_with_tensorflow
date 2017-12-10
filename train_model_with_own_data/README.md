# Train with standard public dataset

## Setup

* All the command should be run in this location `models/research`
* Before run any command, be sure to configure **PYTHONPATH**    

  `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`

## Prepare Input data

Follow this [link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md#generating-the-oxford-iiit-pet-tfrecord-files) to create Oxford-iiit pets data.

This will generates `pet_train.record` and `pet_val.record`. Put them to
`data` folder.

## Prepare other data files

* label file: copy `pet_label_map.pbtxt` to `data` folder.    
    this is the mapping between pet name and ID.

## Prepare model files

* Download `ssd_inception_v2` model from here: [link](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz). Extract files and put them into `models/ssd_inception_v2`


## Prepare pipline config file

Copy `ssd_inception_v2_pets.config` into `models/ssd_inception_v2`

## Run the training

set following variables

```
export PATH_TO_YOUR_PIPELINE_CONFIG=../../train_model_with_own_data/models/ssd_inception_v2/ssd_inception_v2_pets.config

export PATH_TO_TRAIN_DIR=../../train_model_with_own_data/models/ssd_inception_v2/train
```

Run!
```
python object_detection/train.py     --logtostderr     --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG}     --train_dir=${PATH_TO_TRAIN_DIR}
```

## Run the evaluation

set variables
```
export PATH_TO_EVAL_DIR=../../train_model_with_own_data/models/ssd_inception_v2/eval/
```

Run!
```
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}
```
