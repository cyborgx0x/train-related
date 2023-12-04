import torch
import transformers
from finetune_peft import get_peft_config, PEFTArguments
from peft import get_peft_model

model_path = ...
peft_path = ...
tokenizer_path = ...

torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = transformers.LLaMAForCausalLM.from_pretrained(model_path)
peft_config = get_peft_config(peft_args=PEFTArguments(peft_mode="lora"))
model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

tokenizer = transformers.LLaMATokenizer.from_pretrained(tokenizer_path)
batch = tokenizer("The LLaMA language model is", return_tensors="pt")

with torch.no_grad():
    out = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=torch.ones_like(batch["input_ids"]),
        max_length=200,
    )
print(tokenizer.decode(out[0]))