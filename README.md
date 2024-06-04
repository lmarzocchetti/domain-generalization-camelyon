# domain-generalization-camelyon
Method to generalize on Tumor Patches with Camelyon dataset

### Installing Dependencies 
```
pip install -r requirements.txt
```

### Start Training
```
$ python camelyon_triplet.py --root_dataset <path> --download True 
```

Other option can be specified with these arguments:
```
options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch Size for Training
  --num_epochs NUM_EPOCHS
                        Number of Epochs to Train
  --checkpoint_epochs CHECKPOINT_EPOCHS
                        Number of Epochs between Checkpoints
  --learning_rate LEARNING_RATE
                        Learning Rate
  --seed SEED           Random Seed
  --resume_training RESUME_TRAINING
                        Path to Model to Resume Training
  --save_model_path SAVE_MODEL_PATH
                        Path to Save Model (specify with / at the end)
  --root_dataset ROOT_DATASET
                        Path to Root Dataset
  --download DOWNLOAD   Download Dataset
```

