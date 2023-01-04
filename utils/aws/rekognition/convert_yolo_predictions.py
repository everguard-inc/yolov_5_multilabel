import os
import tqdm
import json
import argparse
import os


def get_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("--images", type=str, required=True, help="path to images")
    parser.add_argument("--predicts", type=str, required=True, help="path to predicts")

    return parser.parse_args()

def main(parser: argparse.ArgumentParser) -> None:
    args = get_args(parser)
    ann = os.listdir(args.predicts)
    d = {}
    for i in tqdm(ann):
        name = i.replace("txt","jpg")
        d[name] = []
        img = cv2.imread(args.images+ name)
        with open(args.predicts+i) as f:
            lines = f.readlines()
        for person in lines:
            ok = person.split(" ")
            ok[4] =ok[4].split('\n')[0]
            if int(ok[0])==0:
                label = "with"
            else:
                label = "without"
            w = int(float(ok[3])*img.shape[1])
            h = int(float(ok[4])*img.shape[0])
            x1 = int(float(ok[1])*img.shape[1]-w/2)
            y1 = int(float(ok[2])*img.shape[0]-h/2)
            d[name].append([label, x1,y1,w,h])
    with open('yolo.json', 'w') as outfile:
        json.dump(d, outfile)
        
if __name__ == "__main__":
    main(argparse.ArgumentParser())