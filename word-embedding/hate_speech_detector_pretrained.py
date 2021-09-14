from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from data_preprocessing import read_data


def id2tag(idx):
    id2tag_dict = {
        0: "none",
        1: "offensive",
        2: "hate"
    }
    return id2tag_dict[idx]


print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Constants
# Directories
FINE_TUNE_DIR = "../datasets/Mixed_dataset/"
ROOT_DIR = "../datasets/Competition_dataset/"
TRAIN_DATA = "train.hate.csv"
TEST_DATA = "dev.hate.csv"
TRAIN_DIR = FINE_TUNE_DIR + TRAIN_DATA
TEST_DIR = ROOT_DIR + TEST_DATA
OUTPUT_DIR = "./results"
LOG_DIR = "./logs"

BASE_MODEL = "monologg/koelectra-base-v3-hate-speech"

RESULTS_DIR = "./results/"
CP_DIR = "checkpoint-9500/"
TORCH_MODEL = RESULTS_DIR + CP_DIR

# Trainer Parameter
TRAIN_EPOCHS = 40
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 16
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10

# Read Data
train_data = read_data(TRAIN_DIR)
test_data = read_data(TEST_DIR)

torch.cuda.empty_cache()

# Load Model
model = AutoModelForSequenceClassification.from_pretrained(TORCH_MODEL)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    logging_dir=LOG_DIR,
    logging_steps=LOGGING_STEPS
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data
)

trainer.train(TORCH_MODEL)
