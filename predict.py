import torch

from transformers import BertTokenizer
from PIL import Image
import argparse

from models import caption
from datasets import coco, utils
from configuration import Config
import os
import cv2
import json

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image', required=True)
parser.add_argument('--v', type=str, help='version', default='v3')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
args = parser.parse_args()
image_path = args.path
version = args.v
checkpoint_path = args.checkpoint

config = Config()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
    print("Checking for checkpoint.")
    if checkpoint_path is None:
      raise NotImplementedError('No model to chose from!')
    else:
      if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
      print("Found checkpoint! Loading!")
      model,_ = caption.build_model(config)
      model.to(config.device)
      print("Loading Checkpoint...")
      checkpoint = torch.load(checkpoint_path, map_location='cuda')
      model.load_state_dict(checkpoint['model'])

print("finsiheddddddddddddddddddddddddd")
torch.save({
    'model': model.state_dict(),
}, config.checkpoint)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

f = open("archive/sushi_dict.json")
#f = open("sushi_lid_dataset/annotations/sushi_dict.json")
sushi2id = json.load(f)
# print(sushi2id)
id2sushi ={v:k for k,v in sushi2id.items()}
# print(id2sushi)
f.close()

sushi_types = list(id2sushi.keys())

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


@torch.no_grad()
def evaluate():
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption.to(config.device), cap_mask.to(config.device))
        # print(predictions)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption


if image_path.endswith(".json"):
    import json
    import shutil
    from pathlib import Path
    
    testresult_folder = "testresult"
    Path(testresult_folder).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(testresult_folder)
    Path(testresult_folder).mkdir(parents=True, exist_ok=True)
    Path(testresult_folder+"/FP").mkdir(parents=True, exist_ok=True)
    Path(testresult_folder+"/TP").mkdir(parents=True, exist_ok=True)
    Path(testresult_folder+"/FN").mkdir(parents=True, exist_ok=True)
    Path(testresult_folder+"/TN").mkdir(parents=True, exist_ok=True)

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    with open(image_path) as f:
        data = json.load(f)

    for annotate in data["annotations"]:
        image_path = os.path.join("sushiexpress_dataset", annotate["image_id"]+".jpg")
        #image_path = os.path.join("sushi_lid_dataset", annotate["image_id"]+".jpg")
        captionS = annotate["caption"]
        captionS = id2sushi[captionS]
        
        image = Image.open(image_path)
        image = coco.val_transform(image)
        image = image.unsqueeze(0)
        image = image.to(config.device)

        caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

        output = evaluate()

        result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        result = result.replace(" - ", "-")
        
        if result in id2sushi:
            result = id2sushi[result]
        #result = tokenizer.decode(output[0], skip_special_tokens=True)

        filename = os.path.basename(image_path)

        savefolder = os.path.join(testresult_folder, result)
        Path(savefolder).mkdir(parents=True, exist_ok=True)
        savepath = os.path.join(savefolder, filename)

        if result == captionS and result != "nil":
            TP += 1
            outfile = os.path.join(testresult_folder, "TP", filename)
        elif result == captionS and result == "nil":
            TN += 1
            outfile = os.path.join(testresult_folder, "TN", filename)
        elif result == "nil": 
            FN += 1
            outfile = os.path.join(testresult_folder, "FN", filename)
        else:
            print("false positive")
            print(image_path)
            print("ground truth, ", captionS)
            print(result.capitalize())
            outfile = os.path.join(testresult_folder, "FP", filename)
            FP += 1
        

        image = cv2.imread(image_path)
        cv2.putText(img=image, text=result, org=(10, 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=(0, 0, 255),thickness=1)
        # print(savepath)
        cv2.imwrite(savepath, image)
else:

    image = Image.open(image_path)
    image = coco.val_transform(image)
    image = image.unsqueeze(0)
    image = image.to(config.device)


    caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

    output = evaluate()

    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    #result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(result.capitalize())


    image = cv2.imread(image_path)
    cv2.putText(img=image, text=result, org=(10, 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=(0, 0, 255),thickness=1)
    # show the image, provide window name first
    cv2.imshow('image window', image)
    # add wait key. window waits until user presses a key
    cv2.waitKey(0)
    # and finally destroy/close all open windows
    cv2.destroyAllWindows()


print("TP, TN, FP, FN")
print(TP, TN, FP, FN)
