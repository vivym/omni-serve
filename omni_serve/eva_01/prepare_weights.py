from pathlib import Path

import torch
from transformers import Blip2Model, Blip2Processor

from omni_serve.utils.init import no_init


def main():
    save_path = Path("weights/eva-01-vit")
    if not save_path.exists():
        save_path.mkdir(parents=True)

    with no_init():
        model = Blip2Model.from_pretrained(
            "Salesforce/blip2-flan-t5-xxl",
            torch_dtype=torch.float16,
        )

    model.vision_model.save_pretrained(
        save_path,
        safe_serialization=True,
    )

    processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-flan-t5-xxl"
    )
    processor.save_pretrained(
        save_path,
        safe_serialization=True,
    )


if __name__ == "__main__":
    main()
