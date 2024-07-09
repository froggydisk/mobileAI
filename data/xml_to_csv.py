import os
import random
import shutil
import pandas as pd
import xml.etree.ElementTree as et


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


# xml 파일 내용을 DataFrame에 추가
def process_xml_file(file, df, index):
    tree = et.ElementTree(file=file)
    root = tree.getroot()

    filename = root.find('filename')
    size = root.find('size')
    obj = root.find('object')
    bbox = obj.find('bndbox')

    df.loc[index] = [
        filename.text,  # ImageID: 파일명
        round(int(bbox.find('xmin').text) / \
              int(size.find('width').text), 4),  # XMin: 바운딩 박스 좌표
        round(int(bbox.find('xmax').text) / \
              int(size.find('width').text), 4),  # XMax: 바운딩 박스 좌표
        round(int(bbox.find('ymin').text) / \
              int(size.find('height').text), 4),  # YMin: 바운딩 박스 좌표
        round(int(bbox.find('ymax').text) / \
              int(size.find('height').text), 4),  # XMax: 바운딩 박스 좌표
        obj.find('name').text  # Class명
    ]

    return filename.text


def process_dataset(csv_list, str_list, current_dir):
    for i, csv in enumerate(csv_list):
        image_path = os.path.join(current_dir, str_list[i])
        label_path = os.path.join(current_dir, 'label', str_list[i])

        # 각 데이터셋(image)를 저장할 폴더가 존재하지 않을 시 생성
        create_directory_if_not_exists(image_path)
        # 각 데이터셋(xml)을 저장할 폴더가 존재하지 않을 시 생성
        create_directory_if_not_exists(label_path)

        # 훈련에 필요한 데이터 컬럼 생성
        df = pd.DataFrame(
            columns=['ImageID', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])

        for j, file in enumerate(csv):
            filename = process_xml_file(file, df, j+1)

            shutil.move(file, label_path)  # 각 파일(xml)을 적합한 데이터셋 폴더로 이동
            shutil.move(filename, image_path)  # 각 파일(image)를 적합한 데이터셋 폴더로 이동

        # 각 데이터셋 별로 csv 파일에 저장
        df.to_csv(os.path.join(
            current_dir, f'sub-{str_list[i]}-annotations-bbox.csv'), index=False)


if __name__ == "__main__":
    current_dir = os.getcwd()

    train_ratio = 0.7  # train 데이터 비율
    validation_ratio = 0.1  # validation 데이터 비율
    test_ratio = 0.2  # test 데이터 비율

    file_list = os.listdir(current_dir)
    file_list_xml = [file for file in file_list if file.endswith(
        ".xml")]  # 해당 경로에서 xml 파일만 추출
    random.shuffle(file_list_xml)  # 데이터셋을 임의로 나누기 위한 셔플
    list_len = len(file_list_xml)

    train_csv = file_list_xml[:int(list_len * train_ratio)]
    validation_csv = file_list_xml[int(list_len * train_ratio)
                                :int(list_len * (train_ratio + validation_ratio))]
    test_csv = file_list_xml[int(list_len * (1 - test_ratio)):]
    csv_list = [train_csv, validation_csv, test_csv]
    str_list = ['train', 'validation', 'test']

    create_directory_if_not_exists(os.path.join(current_dir, 'label'))

    process_dataset(csv_list, str_list, current_dir)