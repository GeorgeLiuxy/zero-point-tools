# -*- coding: utf-8 -*-
import oss2
import os

auth = oss2.Auth('<yourAccessKeyId>', '<yourAccessKeySecret>')

bucket = oss2.Bucket(auth, 'http://oss-cn-beijing.aliyuncs.com', 'bbcaudio')

path = os.getcwd()

easy_dir = f"{path}\\bbc_easy"
med_dir = f"{path}\\bbc_medium"

def upload_dir(dir):
    for root, dirs, files in os.walk(dir):
        files.pop(0)
        for file in files:
            item_level = dir.split('\\')[-1].split('_')[-1]
            item_name = f"{item_level}-{file}"
            item_path = f"{dir}\\{file}"
            bucket.put_object_from_file(item_name, item_path)


def main():
    upload_dir(easy_dir)
    upload_dir(med_dir)


if __name__ == '__main__':
    main()
