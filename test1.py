import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

def analyze_data_distribution(dataset):
    print("\nData Analysis:")
    df = pd.DataFrame(dataset)

    print(f"Total number of examples: {len(df)}")

    # Basic statistics about questions
    question_lengths = df['question'].str.len()
    print("\nText length statistics:")
    print(f"Average question length: {question_lengths.mean():.2f} characters")

    # Check references lengths if available
    if 'references' in df.columns:
        # Just take the first reference for a rough measure
        answer_lengths = df['references'].apply(lambda x: len(x[0]) if len(x) > 0 else 0)
        print(f"Average reference answer length: {answer_lengths.mean():.2f} characters")


def prepare_data(tokenizer, train_size=0.8, random_state=42):
    # Load TruthfulQA dataset with "generation" configuration
    ds = load_dataset("truthfulqa/truthful_qa", "generation")
    df = pd.DataFrame(ds['validation'])  # 'validation' split contains the full data in this dataset

    # Create train/validation split
    train_df, valid_df = train_test_split(
        df, train_size=train_size, random_state=random_state
    )

    # Format text: we use 'correct_answers' from generation config as they are equivalent to references
    def format_text(row):
        answer = row['correct_answers'][0] if len(row['correct_answers']) > 0 else ""
        return f"The answer to this question: {row['question']} is : {answer}"

    train_texts = [format_text(row) for _, row in train_df.iterrows()]
    valid_texts = [format_text(row) for _, row in valid_df.iterrows()]

    print(f"Number of training texts: {len(train_texts)}")
    print(f"Number of validation texts: {len(valid_texts)}")
    print("\nSample text:")
    print(train_texts[0][:200] + "...")

    # Tokenize and prepare datasets
    def tokenize_and_prepare(texts):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="pt"
        )
        input_ids = encodings['input_ids'].tolist()
        attention_mask = encodings['attention_mask'].tolist()
        labels = input_ids.copy()  # For language modeling, labels = input_ids

        return Dataset.from_dict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        })

    train_dataset = tokenize_and_prepare(train_texts)
    valid_dataset = tokenize_and_prepare(valid_texts)

    return train_dataset, valid_dataset


def generate_answer(model, tokenizer, device, question):
    model.eval()
    prompt = f"Question: {question}\nAnswer:"
    #prompt =  f"The answer to this question: {question} is :"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Extract only input_ids from the tokenized input
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=50,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Q: {question}\nModel Answer: {answer}\n")


# Load the dataset for analysis
ds = load_dataset("truthfulqa/truthful_qa", "generation")
print("Dataset Analysis:")
analyze_data_distribution(ds['validation'])

# # Initialize tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
# model = GPT2LMHeadModel.from_pretrained('distilgpt2')

from transformers import AutoTokenizer
from mamba2_torch import Mamba2ForCausalLM

device = "cuda"
mamba2_hf_path = "/home/dettrax/PycharmProjects/mamba2-torch/models/mamba2-130m"

model = Mamba2ForCausalLM.from_pretrained(mamba2_hf_path, local_files_only=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(mamba2_hf_path, local_files_only=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Add padding token
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Generate answer before training
test_question = "Where was Barack Obama born?"
print("Before Training:")
generate_answer(model, tokenizer, device, test_question)

# Get datasets
train_dataset, valid_dataset = prepare_data(tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-truthful",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.1,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=6,
    warmup_steps=100,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
)
#
# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Train the model
trainer.train()

# Generate answer after training
print("After Training:")
generate_answer(model, tokenizer, device, test_question)

test_question1 = "Who is the president of the USA?"
generate_answer(model, tokenizer, device, test_question1)

test_question2 = "Who is the prime minister of India?"
generate_answer(model, tokenizer, device, test_question2)

#Who is narendra modi
test_question3 = "Who is narendra modi?"
generate_answer(model, tokenizer, device, test_question3)

