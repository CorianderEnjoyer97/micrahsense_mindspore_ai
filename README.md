# ML Emotion Prediction Project

## Overview
The ML Emotion Prediction Project aims to utilize machine learning techniques to accurately predict emotions based on textual data. This project leverages various algorithms and pre-trained models to analyze text and classify emotions effectively.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributors](#contributors)

## Installation
To get started with this project, clone the repository and install the required dependencies. Use the following commands:

```bash
# Clone the repository
$ git clone https://github.com/CorianderEnjoyer97/micrahsense_mindspore_ai.git

# Change directory to the project
$ cd micrahsense_mindspore_ai

# Install required dependencies
$ pip install -r requirements.txt
```

## Usage
After installation, you can utilize the provided scripts to train and evaluate the emotion prediction model. 

### Training the Model
To train the model, run:
```bash
$ python train.py
```

### Predicting Emotion
To predict emotions from new text data, run:
```bash
$ python predict.py --input "Your text here"
```

## Data
The dataset used for training the model can be found in the `data` directory. Ensure to explore the different files for understanding the dataset structure and format.

### Dataset Information
- **Source:** [Dataset Source Link]
- **Format:** CSV file containing text and corresponding emotion labels.
- **Classes:** Happy, Sad, Angry, Surprised, etc.

## Model Training
The project implements various models using libraries like MindSpore. You can adjust parameters within the `config.py` file to tune the models according to your needs.

## Evaluation
After training, the model can be evaluated using metrics such as accuracy, precision, and recall. Refer to the evaluation script to see how these metrics are calculated.

### Running the Evaluation
To evaluate the model's performance, run:
```bash
$ python evaluate.py
```

## Contributors
- **CorianderEnjoyer97** - Initial work


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.