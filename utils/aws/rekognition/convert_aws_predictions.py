import os
import tqdm
import json
import argparse
import os


def get_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("--images", type=str, required=True, help="path to images")
    parser.add_argument("--predicts", type=str, required=True, help="path to images")

    return parser.parse_args()

def main(parser: argparse.ArgumentParser) -> None:
    args = get_args(parser)
    ann = os.listdir(args.predicts)
    d = {}
    for i in tqdm(ann):
        name = i.replace("json","jpg")
        d[name] = []
        img = cv2.imread(args.predicts+ name)
        with open('aws_results/{}'.format(i)) as json_file:
            data = json.load(json_file)
            for person in data["Persons"]:
                print(person["Confidence"])
                if "HEAD_COVER" in str(person):
                    label = "with"
                else:
                    label = "without"
                box = person["BoundingBox"]
                w = int(box["Width"]*img.shape[1])
                h = int(box["Height"]*img.shape[0])
                x1 = int(box["Left"]*img.shape[1])
                y1 = int(box["Top"]*img.shape[0])
                d[name].append([label, x1,y1,w,h])
    with open('aws.json', 'w') as outfile:
        json.dump(d, outfile)
        
if __name__ == "__main__":
    main(argparse.ArgumentParser())