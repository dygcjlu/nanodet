from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import shutil
import json

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
              'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
              'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
              'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
              'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

def exchact_person_dog_cat():
    phase = 'test'
    phase_root = phase + '2017'
    ann_file = 'instances_' + phase_root + '.json'
    img_path = 'D:\\deng\\dataset\\coco\\2017\\' + phase_root + '\\'
    #ann_path = 'D:\\deng\\dataset\\coco\\2017\\annotations\\' + ann_file
    # img_path = 'D:\\deng\\dataset\\coco\\2017\\val2017\\'
    ann_path = 'D:\\deng\\dataset\\coco\\2017\\annotations\\image_info_test2017.json'
    coco_api = COCO(ann_path)

    new_dataset = {}
    info = {}  # 存放字典
    licenses = []  # 存放字典列表
    images = []  # 存放字典列表
    annotations = []  # 存放字典列表
    categories = []

    info = coco_api.dataset['info']
    new_dataset['info'] = info

    licenses = coco_api.dataset['licenses']
    new_dataset['licenses'] = licenses

    print(coco_api.getCatIds())
    target_object = ['person', 'dog', 'cat']

    is_person = False
    count = 0
    drop_out_interval = 7

    for target_name in target_object:
        class_count = 0
        target_list = []
        # target_name = 'dog'
        if target_name == 'person':
            is_person = True
        else:
            is_person = False
        target_id = 0
        # target_path = 'D:\\deng\\dataset\\coco\\mine\\' + target_name + '\\'
        target_path = 'D:\\deng\\dataset\\coco\\mine\\' + phase + '\\person_cat_dog\\'
        for i in range(1, len(coco_api.cats)):
            if i in coco_api.cats:
                cat = coco_api.cats[i]
                if cat['name'] == target_name:
                    target_id = cat['id']
                    categories.append(cat)

        target_list = coco_api.getImgIds(catIds=[target_id])
        print(target_name, len(target_list))
        for target_id in target_list:
            count += 1
            if is_person and count < 6:
                continue  # person相关图片有64115，cat有4114，dog有4385,需要把person丢掉一些
            else:
                count = 0

            class_count += 1
            if target_id in coco_api.imgs:
                ann_ids = coco_api.getAnnIds([target_id])
                anns = coco_api.loadAnns(ann_ids)
                for ann in anns:
                    annotations.append(ann)

                img = coco_api.imgs[target_id]
                images.append(img)
                file_name = img['file_name']
                src_img_name = os.path.join(img_path, file_name)
                dst_img_name = os.path.join(target_path, file_name)
                shutil.copy(src_img_name, dst_img_name)
        print('saved image count:', class_count)

    new_dataset['images'] = images
    new_dataset['annotations'] = annotations
    new_dataset['categories'] = categories

    # ann_file = 'D:\\deng\\dataset\\coco\\mine\\train\\' + target_name + '.json'
    ann_file = 'D:\\deng\\dataset\\coco\\mine\\' + phase + '\\person_cat_dog_test.json'

    with open(ann_file, 'w') as f:
        json.dump(new_dataset, f)

    # img = cv2.imread(image_path)

    # rint(len(person_list))
    # print(person_list[0])

def coco_eval():
    path_to_annotation = r'D:\deng\dataset\coco\mine\val\person_cat_dog_val.json'

    cocoGt = COCO(path_to_annotation)
    result = r'D:\deng\work\pet\model\person_dog_cat\efficient0_320_deng\results0.json'
    cocoDt = cocoGt.loadRes(result)
    annType = 'bbox'
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.evaluate()
    cocoEval.accumulate()

    #original_stdout = sys.stdout
    #string_stdout = StringIO()
    #sys.stdout = string_stdout
    cocoEval.summarize()
    #sys.stdout = original_stdout

    mean_ap = cocoEval.stats[0].item()  # stats[0] records AP@[0.5:0.95]
    #detail = string_stdout.getvalue()
    return mean_ap#, detail

if __name__ == '__main__':
    exchact_person_dog_cat()
    #coco_eval()
