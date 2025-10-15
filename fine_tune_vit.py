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



dataset = load_dataset('./memes')

# ===============================
# 2. Preprocess images
# ===============================
model_id = "apple/mobilevit-small"
processor =  MobileViTImageProcessor.from_pretrained(model_id)

def transform(example_batch):
    images = []
    valid_labels = []
    
    for img, label in zip(example_batch['image'], example_batch['label']):
        try:
         
            
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
            else:
                raise ValueError("Image is not a valid PIL Image or path")
            
            images.append(img)
            valid_labels.append(label)
        except Exception as e:
            print(f"Skipping bad image due to error: {e}")
            continue

    # Only process valid images
    inputs = processor(images, return_tensors='pt')
    inputs['label'] = valid_labels
    
    return inputs

dataset = dataset.with_transform(transform)



model = MobileViTForImageClassification.from_pretrained(
    model_id,
    num_labels=2,
    ignore_mismatched_sizes=True  # replaces classifier head
)


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch])
    }


metric = load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)



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


trainer.train()


trainer.save_model("./mobilevit-small-finetuned")
processor.save_pretrained("./mobilevit-small-finetuned")

