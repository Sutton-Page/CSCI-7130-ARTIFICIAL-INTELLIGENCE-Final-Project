# Hateful Memes Classification  

This project trains a model on the [Hateful Memes dataset](https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset/data) using TensorFlow.  

---

## Setup Instructions  


### 1. Download the Hateful Memes Dataset  
- Download the dataset and extract the zip file.  
- You should have a folder named `data`.  

### 2. Clone This Repository  

- Clone this repo to your computer.
- Then move the unziped folder `data` containing the hateful meme dataset to the folder containing the repo you cloned or downloaded.

### 3. Installing Dependencies

Open a command prompt in the folder for this repo and run the command 

```bash
pip install -r requirements.txt
```

To install all the required dependencies.

### 4. Creating dataset for tensorflow

Within the folder for this repo run the command
```bash
python create_dataset.py
```
This file will take a minute to run as it is moving a bunch of files to new folders.
### 5. Training the dataset
- Run the command below to train the model on the hateful meme dataset.
```bash
python train_model_with_dataset.py
```

