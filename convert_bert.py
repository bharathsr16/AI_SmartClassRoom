import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from pathlib import Path
import torch
from openvino.tools.mo import convert_model
import openvino as ov

# ✅ Use smaller, faster QA model for testing
model_name = "distilbert-base-uncased-distilled-squad"

# 🔽 Download model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 🧠 Dummy inputs for OpenVINO conversion
question = "What is photosynthesis?"
context = (
    "Photosynthesis is the process by which green plants and some other organisms "
    "use sunlight to synthesize foods from carbon dioxide and water."
)
inputs = tokenizer(question, context, return_tensors="pt", max_length=128, padding="max_length", truncation=True)

# 🔁 Convert PyTorch model to OpenVINO IR format
ov_model = convert_model(model, example_input=tuple(inputs.values()))

# 💾 Save to bert_ir/
output_dir = Path("bert_ir")
output_dir.mkdir(parents=True, exist_ok=True)
ov.save_model(ov_model, output_dir / "distilbert_qa.xml")

print("✅ Model converted and saved to: bert_ir/distilbert_qa.xml")
