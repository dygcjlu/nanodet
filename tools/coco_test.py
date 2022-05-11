from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import shutil
import json
import cv2 as cv

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


def exchact_person_dog_cat_v2():
    img_path = '/root/deng/dataset/coco/mine/person_dog_cat_v2/total'
    target_path = ''
    ann_path = '/root/deng/dataset/coco/mine/person_dog_cat_v2/instances_train2017.json'
    target_ann_file = '/root/deng/dataset/coco/mine/person_dog_cat_v2/person_dog_cat_train_val.json'
    coco_api = COCO(ann_path)
    img_name2id = {}
    for img in coco_api.dataset['images']:
        img_name2id[img["file_name"]] = img["id"]


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
    #catIds = [1, 17, 18]
    catIds = []

    for i in range(1, len(coco_api.cats)):
        if i in coco_api.cats:
            cat = coco_api.cats[i]
            if cat['name'] in target_object:
                categories.append(cat)
                catIds.append(cat['id'])

    file_list = os.listdir(img_path)
    print('total:' + str(len(file_list)))
    handdled = 0
    for i, file in enumerate(file_list):
        img_id = img_name2id[file]
        annIds = coco_api.getAnnIds(imgIds=img_id)
        #imgInfo = coco_api.loadImgs(img_id)[0]

        for annId in annIds:
            annInfo = coco_api.loadAnns(annId)[0]
            if annInfo["category_id"] not in catIds:
                continue
            annotations.append(annInfo)

        img = coco_api.imgs[img_id]
        images.append(img)
        file_name = img['file_name']
        src_img_name = os.path.join(img_path, file_name)
        dst_img_name = os.path.join(target_path, file_name)
        #shutil.copy(src_img_name, dst_img_name)
        handdled += 1
        #shutil.move(src_img_name, dst_img_name)

    new_dataset['images'] = images
    new_dataset['annotations'] = annotations
    new_dataset['categories'] = categories

    print('handled:' + str(handdled))

    with open(target_ann_file, 'w') as f:
        json.dump(new_dataset, f)




#按2 比8分val 和 trainl
def split_train_val():
    img_path = '/root/deng/dataset/coco/mine/person_dog_cat_v2/total'
    ann_path = '/root/deng/dataset/coco/mine/person_dog_cat_v2/person_dog_cat_train_val.json'
    ann_path_train = '/root/deng/dataset/coco/mine/person_dog_cat_v2/person_dog_cat_train.json'
    ann_path_val = '/root/deng/dataset/coco/mine/person_dog_cat_v2/person_dog_cat_val.json'
    coco_api = COCO(ann_path)

    categories = []

    new_dataset_train = {}
    images_train = []  # 存放字典列表
    annotations_train = []  # 存放字典列表

    new_dataset_val = {}
    images_val = []  # 存放字典列表
    annotations_val = []  # 存放字典列表

    print(coco_api.getCatIds())
    #catIds = coco_api.getCatIds()
    catIds = [18, 17, 1]

    for i in catIds:
        cat = coco_api.cats[i]
        categories.append(cat)

    img_ids = []


    for catId in catIds:
        imgIds = coco_api.getImgIds(catIds=catId)

        val_th = 5
        img_count = 0
        val_count = 0
        train_count = 0
        skip_count = 0
        for imgId in imgIds:
            if imgId in img_ids:
                # this image has been handled by other catID
                skip_count += 1
                continue

            img_ids.append(imgId)
            img_count += 1
            annIds = coco_api.getAnnIds(imgIds=imgId)

            for annId in annIds:
                annInfo = coco_api.loadAnns(annId)[0]
                if img_count % val_th == 0:
                    annotations_val.append(annInfo)
                else:
                    annotations_train.append(annInfo)

            img = coco_api.imgs[imgId]
            if img_count % val_th == 0:
                val_count += 1
                images_val.append(img)
            else:
                train_count += 1
                images_train.append(img)
        print('catId:%d, total:%d, val:%d, train:%d, skip_count:%d'%(catId, img_count, val_count, train_count, skip_count))

    new_dataset_val['images'] = images_val
    new_dataset_val['annotations'] = annotations_val
    new_dataset_val['categories'] = categories

    with open(ann_path_val, 'w') as f:
        json.dump(new_dataset_val, f)

    new_dataset_train['images'] = images_train
    new_dataset_train['annotations'] = annotations_train
    new_dataset_train['categories'] = categories

    with open(ann_path_train, 'w') as f:
        json.dump(new_dataset_train, f)

    print('end')

#get part of val
def extract_sub_val():
    img_path = '/root/deng/dataset/coco/mine/person_dog_cat_v2/val'
    ann_path = '/root/deng/dataset/coco/mine/person_dog_cat_v2/person_dog_cat_val.json'
    target_path = '/root/deng/dataset/coco/mine/person_dog_cat_v2/sub_val1'
    #ann_path_train = '/root/deng/dataset/coco/mine/person_dog_cat_v2/person_dog_cat_train.json'
    ann_path_val = '/root/deng/dataset/coco/mine/person_dog_cat_v2/person_dog_cat_sub_val1.json'
    coco_api = COCO(ann_path)

    categories = []
    new_dataset_train = {}
    images_train = []  # 存放字典列表
    annotations_train = []  # 存放字典列表
    new_dataset_val = {}
    images_val = []  # 存放字典列表
    annotations_val = []  # 存放字典列表

    print(coco_api.getCatIds())
    #catIds = coco_api.getCatIds()
    catIds = [18, 17, 1]
    for i in catIds:
        cat = coco_api.cats[i]
        categories.append(cat)
    img_ids = []

    for catId in catIds:
        imgIds = coco_api.getImgIds(catIds=catId)

        val_th = 5
        img_count = 0
        val_count = 0
        train_count = 0
        skip_count = 0
        for imgId in imgIds:
            if imgId in img_ids:
                # this image has been handled by other catID
                skip_count += 1
                continue
            img_ids.append(imgId)
            img_count += 1
            annIds = coco_api.getAnnIds(imgIds=imgId)

            for annId in annIds:
                annInfo = coco_api.loadAnns(annId)[0]
                if img_count < 65:
                    annotations_val.append(annInfo)

            img = coco_api.imgs[imgId]
            if img_count < 65:
                val_count += 1
                images_val.append(img)
                file_name = img['file_name']
                src_img_name = os.path.join(img_path, file_name)
                dst_img_name = os.path.join(target_path, file_name)
                shutil.copy(src_img_name, dst_img_name)

        print('img count:%d_%d'% (img_count, val_count))

    new_dataset_val['images'] = images_val
    new_dataset_val['annotations'] = annotations_val
    new_dataset_val['categories'] = categories
    with open(ann_path_val, 'w') as f:
        json.dump(new_dataset_val, f)
    print('end')

def copy_train_val_image():
    img_path = '/root/deng/dataset/coco/mine/person_dog_cat_v2/total'
    target_path = '/root/deng/dataset/coco/mine/person_dog_cat_v2/val'

    ann_path = '/root/deng/dataset/coco/mine/person_dog_cat_v2/person_dog_cat_val.json'
    #ann_path_val = '/root/deng/dataset/coco/mine/person_dog_cat_v2/person_dog_cat_val.json'
    coco_api = COCO(ann_path)

    catIds = coco_api.getCatIds()
    img_list = []
    repeated_img = 0

    for catId in catIds:
        imgIds = coco_api.getImgIds(catIds=catId)
        for imgId in imgIds:
            if imgId in img_list:
                repeated_img+=1
                continue
            img = coco_api.imgs[imgId]
            file_name = img['file_name']
            src_img_name = os.path.join(img_path, file_name)
            dst_img_name = os.path.join(target_path, file_name)
            shutil.copy(src_img_name, dst_img_name)
            img_list.append(imgId)

    print('repeated:%d'%(repeated_img))




"""
def coco_eval(): # it is wrong
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
"""
def visualize_coco():
    phase = 'test'
    phase_root = phase + '2017'
    ann_file = 'instances_' + phase_root + '.json'
    img_path = '/root/deng/dataset/coco/mine/val/person_cat_dog/'
    # ann_path = 'D:\\deng\\dataset\\coco\\2017\\annotations\\' + ann_file
    # img_path = 'D:\\deng\\dataset\\coco\\2017\\val2017\\'
    ann_path = '/root/deng/dataset/coco/mine/val/person_cat_dog_val.json'
    coco_api = COCO(ann_path)

    print(coco_api.getCatIds())

    catIds = coco_api.getCatIds()

    for catId in catIds:
        imgIds = coco_api.getImgIds(catIds=catId)
        for imgId in imgIds:
            annIds = coco_api.getAnnIds(imgIds=imgId)
            imgInfo = coco_api.loadImgs(imgId)[0]
            print(imgInfo)
            img_file = img_path + imgInfo["file_name"]
            img = cv.imread(img_file)

            for annId in annIds:
                annInfo = coco_api.loadAnns(annId)[0]
                if annInfo["category_id"] != catId:
                    continue
                bbox = annInfo["bbox"]
                x = int(bbox[0])
                y = int(bbox[1])
                #w = int(bbox[2] - bbox[0])
                #h = int(bbox[3] - bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                if catId == 1:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                elif catId == 17:
                    cv.rectangle(img,  (x, y), (x + w, y + h), (0, 255, 0), 1)
                elif catId == 18:
                    cv.rectangle(img,  (x, y), (x + w, y + h), (255, 0, 0), 1)
                if annInfo["iscrowd"] == 1:
                    cv.rectangle(img, (x, y), (x + w, y + h), (50, 50, 150), 2)

            cv.imshow("img", img)
            cv.waitKey(800)



def phone_filter():
    src_path = '/root/deng/qualityInspection/dataset/origin_phone/'
    target_path = '/root/deng/qualityInspection/dataset/za/'

    for path, currentDirectory, files in os.walk(src_path):
        for file in files:
            print(os.path.join(path, file))


if __name__ == '__main__':
    #exchact_person_dog_cat()
    #coco_eval()
    visualize_coco()
    #phone_filter()
    #exchact_person_dog_cat_v2()
    #extract_sub_val()
    #split_train_val()
    #copy_train_val_image()
