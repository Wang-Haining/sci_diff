#!/usr/bin/env python3
import torch
import torch.nn.functional as functional
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

BASE_REVISION = "3447645e1def9117997203454fa4495937bfbd83"
ADAPTER_REVISION = "2081559630a80fc5851d8f798a05ba81e9468089"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("expected an allocated CUDA GPU, got torch.cuda.is_available()=False")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base", revision=BASE_REVISION)
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base", revision=BASE_REVISION)
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True,
                       revision=ADAPTER_REVISION)
    if list(model.active_adapters.flatten()) != ["specter2"]:
        raise RuntimeError(f"expected active SPECTER2 adapter, got {model.active_adapters}")
    model.eval().cuda()
    text = ["Journal specialization and scientific diffusion",
            "Clinical outcomes after cardiac surgery" + tokenizer.sep_token +
            "We estimate postoperative outcomes in a multicenter cohort."]
    tokens = tokenizer(text, padding=True, truncation=True, max_length=512,
                       return_tensors="pt", return_token_type_ids=False).to("cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        embedding = functional.normalize(model(**tokens).last_hidden_state[:, 0, :].float(), dim=1)
    norms = embedding.norm(dim=1).cpu()
    if embedding.shape != (2, 768) or not torch.allclose(norms, torch.ones(2), atol=1e-4):
        raise ValueError(f"SPECTER2 smoke QC failed: shape={tuple(embedding.shape)}, norms={norms.tolist()}")
    print({"device": torch.cuda.get_device_name(), "shape": tuple(embedding.shape),
           "norms": norms.tolist(), "cosine": float(embedding[0] @ embedding[1]),
           "base_commit": BASE_REVISION, "adapter_commit": ADAPTER_REVISION})


if __name__ == "__main__":
    main()
