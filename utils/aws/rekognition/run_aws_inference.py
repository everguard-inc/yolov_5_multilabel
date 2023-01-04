import boto3
import json
from tqdm import tqdm
import argparse
import os


def get_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("--images", type=str, required=True, help="path to images")
    parser.add_argument("--save-json", type=str, required=True, help="where to save jsons")

    return parser.parse_args()

def main(parser: argparse.ArgumentParser) -> None:
    args = get_args(parser)
    images = os.listdir(args.images)
    rekognition = boto3.client('rekognition', region_name='us-west-2')
    for img in tqdm(images):
        with open(args.images+img, 'rb') as document:
            imageBytes = bytearray(document.read())


        # Call Amazon Rekognition
        response = rekognition.detect_protective_equipment(
                Image={'Bytes': imageBytes},
                SummarizationAttributes={
                    'MinConfidence': 50,
                    'RequiredEquipmentTypes': [
                        #'FACE_COVER',
                        'HEAD_COVER',
                        #'HAND_COVER',
                    ]
                }
            )

        with open("{}/{}".format(args.save_json, img.replace("jpg", "json")), "w") as f:
            f.write(json.dumps(response))
            
if __name__ == "__main__":
    main(argparse.ArgumentParser())