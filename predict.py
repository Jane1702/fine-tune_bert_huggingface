from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

model = DistilBertForSequenceClassification.from_pretrained("./results/checkpoint-375")
tokenizer = DistilBertTokenizer.from_pretrained("./results/checkpoint-375")

text = "The movie was fantastic!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
outputs = model(**inputs)
logits = outputs.logits
predictions = logits.argmax(-1)
print(f"Result: {predictions.item()}") 
