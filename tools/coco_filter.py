from pycocotools.coco import COCO
import os
import shutil
import json
import cv2 as cv

def main():

    img_path = '/root/deng/dataset/coco/mine/others/person/'
    ann_path = '/root/deng/dataset/coco/mine/others/person.json'
    coco_api = COCO(ann_path)

    save_path = "/root/deng/dataset/coco/mine/others/save/"
    delete_path = "/root/deng/dataset/coco/mine/others/delete/"
    candidate_path = "/root/deng/dataset/coco/mine/others/candidate/"
    indoor_path = "/root/deng/dataset/coco/mine/others/indoor/"

    file_list = os.listdir(img_path)
    catIds = [1]

    img_name2id = {}
    for img in coco_api.dataset['images']:
        img_name2id[img["file_name"]] = img["id"]

    catId = 1
    file_num = len(file_list)
    #i = 0
    for i, file in enumerate(file_list):
        img_id = img_name2id[file]
        annIds = coco_api.getAnnIds(imgIds=img_id)
        imgInfo = coco_api.loadImgs(img_id)[0]

        img_file = img_path + imgInfo["file_name"]
        img = cv.imread(img_file)

        for annId in annIds:
            annInfo = coco_api.loadAnns(annId)[0]
            if annInfo["category_id"] != catId:
                continue
            bbox = annInfo["bbox"]
            x = int(bbox[0])
            y = int(bbox[1])
            # w = int(bbox[2] - bbox[0])
            # h = int(bbox[3] - bbox[1])
            w = int(bbox[2])
            h = int(bbox[3])
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 200), 1)
            if annInfo["iscrowd"] == 1:
                cv.rectangle(img, (x, y), (x + w, y + h), (50, 50, 150), 2)

        # text
        text = 'w-indoor, s-save'
        y1 = 0
        # Using cv2.putText() method
        y1 += 30
        cv.putText(img, text, (10, y1), cv.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 1, cv.LINE_AA, False)

        text = 'd-delete, e-candidate'
        y1 += 30
        cv.putText(img, text, (10, y1), cv.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 0, 255), 1, cv.LINE_AA, False)

        text2 = 'total: ' + str(file_num) + ', ' + str(i) + ' handled'
        y1 += 30
        cv.putText(img, text2, (10, y1), cv.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 0, 255), 1, cv.LINE_AA, False)

        cv.imshow("img", img)
        key_code = cv.waitKey(0)
        if key_code == ord('w'):
            dst_path = indoor_path + imgInfo["file_name"]
            shutil.move(img_file, dst_path)
            print('image %d move to:%s' % (i, dst_path))

        elif key_code == ord('s'):
            dst_path = save_path + imgInfo["file_name"]
            shutil.move(img_file, dst_path)
            print('image %d move to:%s'% (i, dst_path))
        elif key_code == ord('d'):
            dst_path = delete_path + imgInfo["file_name"]
            shutil.move(img_file, dst_path)
            print('image %d move to:%s' % (i, dst_path))
        elif key_code == ord('e'):
            dst_path = candidate_path + imgInfo["file_name"]
            shutil.move(img_file, dst_path)
            print('image %d move to:%s' % (i, dst_path))
        else:

            text = 'unknown operation, set candidate by default'
            print(text)
            y1 += 30
            cv.putText(img, text, (10, y1), cv.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 0, 255), 1, cv.LINE_AA, False)
            cv.imshow("img", img)
            cv.waitKey(1000)


# filter small and crowd object
def coco_filter():
    img_path = '/root/deng/dataset/coco/mine/others/person/'
    ann_path = '/root/deng/dataset/coco/mine/others/person.json'
    coco_api = COCO(ann_path)
    file_list = os.listdir(img_path)
    catIds = [1]

    img_name2id = {}
    for img in coco_api.dataset['images']:
        img_name2id[img["file_name"]] = img["id"]

    catId = 1
    file_num = len(file_list)
    # i = 0
    deleted_num = 0
    for i, file in enumerate(file_list):
        need_delete = False
        img_id = img_name2id[file]
        annIds = coco_api.getAnnIds(imgIds=img_id)
        imgInfo = coco_api.loadImgs(img_id)[0]
        img_file = img_path + imgInfo["file_name"]

        w = imgInfo['width']
        h = imgInfo['height']
        longger = 0
        if w > h:
            longger = w
        else:
            longger =h
        scale = 320.0 / longger

        delete_path = '/root/deng/dataset/coco/mine/others/pre_delete/'
        for annId in annIds:
            annInfo = coco_api.loadAnns(annId)[0]
            if annInfo["category_id"] != catId:
                continue
            is_crowd = annInfo['iscrowd']
            area = annInfo['area'] * scale # scale to 320
            if is_crowd == 1:
                need_delete = True
                break
            if area < (32 * 32):
                need_delete = True
                break

        if need_delete:
            deleted_num += 1
            dst_path = delete_path + imgInfo["file_name"]
            shutil.move(img_file, dst_path)

    print('total:%d, deleted:%d'%(file_num, deleted_num))


def img_test():
    img = cv.imread('/root/deng/dataset/coco/mine/others/za/save/000000375709.jpg')

    dst = cv.resize(img, (320,320))

    x = 100
    y = 100
    w = 32
    h = 32

    cv.rectangle(dst, (x, y), (x + w, y + h), (255, 0, 0), 1)
    cv.imshow('img', dst)
    cv.waitKey(0)

if __name__ == "__main__":
    main()
    #coco_filter()
    #img_test()
