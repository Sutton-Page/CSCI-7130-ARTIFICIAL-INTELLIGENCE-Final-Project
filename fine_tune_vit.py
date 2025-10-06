import torch
import os
from PIL import Image
from datasets import load_dataset,ClassLabel
from transformers import (
    MobileViTImageProcessor,
    MobileViTForImageClassification,
    TrainingArguments,
    Trainer
)
from evaluate import load


# ===============================
# 1. Load dataset
# ===============================
# Your dataset should have train.jsonl, test.jsonl and img/ folder with images.
# Example train.jsonl entry:
# {"image": "img/16395.png", "label": 0}
dataset = load_dataset('./data')

# ===============================
# 2. Preprocess images
# ===============================
model_id = "apple/mobilevit-small"
processor =  MobileViTImageProcessor.from_pretrained(model_id)

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['label'] = example_batch['label']
    return inputs

dataset = dataset.with_transform(transform)
# ===============================
# 3. Load model
# ===============================


model = MobileViTForImageClassification.from_pretrained(
    model_id,
    num_labels=2,
    ignore_mismatched_sizes=True  # replaces classifier head
)

# ===============================
# 4. Data collator
# ===============================
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch])
    }


metric = load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

# ===============================
# 5. Training setup
# ===============================
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    save_steps=30,
    eval_steps=30,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    remove_unused_columns=False,
    push_to_hub=False,
    fp16=torch.cuda.is_available(),  # enable mixed precision on GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,  # processor works as tokenizer here
    data_collator=collate_fn,
)

# ===============================
# 6. Train
# ===============================
trainer.train()

# ===============================
# 7. Save model
# ===============================
trainer.save_model("./mobilevit-small-finetuned")
processor.save_pretrained("./mobilevit-small-finetuned")

