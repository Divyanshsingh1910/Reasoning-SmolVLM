import argparse
import os, json
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
IMAGE_DIR = "../../data/filtered_images_5k/"
test_file = "test_set_answers.json" 


def main(
    checkpoint_path: str,
    output_path: str,
    DEVICE: str = "cuda",
    think_prompt: bool = False,
):

    processor = AutoProcessor.from_pretrained(checkpoint_path)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16, device_map= DEVICE
    )



    with open(test_file, "r") as f:
        test_data = json.load(f)



    MODEL_OUTPUT = []

    for i, item in enumerate(test_data):
        imgpath = IMAGE_DIR + item["imgname"]
        qs = item["query"]
        if think_prompt:
            qs = qs + "\n Let's think step by step.\n"

        try:
            image = Image.open(imgpath).convert("RGB")
        except Exception as e:
            print(f"Error opening image {imgpath}: {e}")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{qs}"}
                ]
            },
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        with torch.no_grad():
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = inputs.to(DEVICE)
            
            generated_ids = model.generate(**inputs
                                        , max_new_tokens = 256
                                        )
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            
            )
        item["output"] = generated_texts[0].split("Assistant:")[1].strip()
        MODEL_OUTPUT.append(item)

        if i%25 == 0:
            print(f"Processed {i} queries")


    print(f"Length of model output: {len(MODEL_OUTPUT)}")

    # if output_path exits, don't overwrite, change the output file name
    if os.path.exists(output_path):
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{len(checkpoint_path)}{ext}"        

    with open(output_path, "w") as f:
        json.dump(MODEL_OUTPUT, f, indent=4)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # take args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--think_prompt", type=bool, default=False)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    output_path = args.output_path
    DEVICE = args.device
    think_prompt = args.think_prompt

    main(checkpoint_path, output_path, DEVICE, think_prompt)   
    
# example usage:
# python eval.py --checkpoint_path /path/to/checkpoint --output_path /path/to/output.json --device cuda --think_prompt True