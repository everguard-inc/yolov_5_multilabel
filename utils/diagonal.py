import numpy as np
import cv2


def get_diagonal_array(feature_map):
    # Extracts the diagonals from the feature map.
    # k=1 and k=-1 refer to the diagonals just above and below the main diagonal.
    diag_k_plus_1 = np.diag(feature_map, k=1)
    diag_k_minus_1 = np.diag(feature_map, k=-1)
    diag_k = np.diag(feature_map)

    combined_diagonals = np.mean(np.concatenate((diag_k_minus_1, diag_k_plus_1,diag_k)))
    return combined_diagonals

def get_diagonal(feature_map, bboxes, lifted_load_label=2, upscale_width=736, upscale_height=416):
    lifted_load_diagonals = []
    upscale_size = (upscale_width, upscale_height)
    for i in bboxes[0]:
        if int(i[-1]) == lifted_load_label:
            x1, y1, x2, y2 = i[:4]
            
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            
            feature_map = feature_map.cpu().numpy()
            feature_map = cv2.resize(feature_map, (upscale_size))
            feature_map = feature_map[y1:y2, x1:x2]
            
            # Gets the diagonal values from left-right and right-left.
            lr = get_diagonal_array(feature_map)
            rl = get_diagonal_array(np.fliplr(feature_map))
            
            return np.abs(lr) > np.abs(rl)
                