import rarfile

with rarfile.RarFile("/Users/liuyuhan/fall-detection-multimodal/data/f_mask_b_1_keypoints_csv.rar") as rf:
    rf.extractall(path= "/Users/liuyuhan/fall-detection-multimodal/data")