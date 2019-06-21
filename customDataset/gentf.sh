#! /bin/bash
# every step to create the tensorflow .record files

# first of all, resize the images to a better size by using resize.sh
# label all the images using labelImg
# then execute the steps below


activate
python xml_to_csv.py
python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=test/
python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=train/
