import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk


# 图像目标计数应用类
class TargetCountingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像目标计数应用")
        self.create_tabs()
        self.model_resnet = ResNet50(weights="imagenet")
        self.model_yolo = YOLO("yolov5s.pt")

    def create_tabs(self):
        notebook = ttk.Notebook(self.root)
        self.tab_app = ttk.Frame(notebook)
        self.tab_data = ttk.Frame(notebook)
        notebook.add(self.tab_app, text="应用")
        notebook.add(self.tab_data, text="数据")
        notebook.pack(expand=1, fill="both")

        # 应用选项卡
        ttk.Label(self.tab_app, text="输入搜索内容:").grid(row=0, column=0)
        self.input_entry = ttk.Entry(self.tab_app, width=30)
        self.input_entry.grid(row=0, column=1)
        self.start_button = ttk.Button(self.tab_app, text="图像目标计数", command=self.start_processing)
        self.start_button.grid(row=1, column=1)

        # 数据选项卡
        self.result_tree = ttk.Treeview(self.tab_data, columns=("类别", "数量"), show="headings")
        self.result_tree.heading("类别", text="类别")
        self.result_tree.heading("数量", text="数量")
        self.result_tree.pack(expand=1, fill="both")

    def start_processing(self):
        search_term = self.input_entry.get()
        if not search_term:
            messagebox.showwarning("警告", "请输入搜索内容！")
            return

        # 步骤 1：网络爬虫获取图片
        messagebox.showinfo("信息", f"开始爬取图片：{search_term}")
        image_dir = "images"
        self.download_images(search_term, save_dir=image_dir)

        # 步骤 2：图像分类
        messagebox.showinfo("信息", "开始进行图像分类...")
        classified_dir = "classified_images"
        self.classify_images(image_dir, classified_dir)

        # 步骤 3：目标检测
        messagebox.showinfo("信息", "开始目标检测...")
        detection_results = self.detect_objects(classified_dir)

        # 步骤 4：显示结果
        self.update_results(detection_results)
        messagebox.showinfo("信息", "处理完成！")

    # 网络爬虫模块
    def download_images(self, search_term, num_images=20, save_dir="images"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        search_url = f"https://www.google.com/search?tbm=isch&q={search_term}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        image_tags = soup.find_all("img")

        count = 0
        for img_tag in image_tags:
            if count >= num_images:
                break
            img_url = img_tag.get("src")
            try:
                img_data = requests.get(img_url).content
                with open(os.path.join(save_dir, f"{search_term}_{count}.jpg"), "wb") as f:
                    f.write(img_data)
                count += 1
            except:
                continue
        print(f"下载完成：{count} 张图片")

    # 图像分类模块
    def classify_images(self, image_dir, output_dir="classified_images"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for img_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_name)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

            preds = self.model_resnet.predict(img_array)
            label = decode_predictions(preds, top=1)[0][0][1]  # 获取标签

            label_dir = os.path.join(output_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            shutil.move(img_path, os.path.join(label_dir, img_name))

    # 目标检测模块
    def detect_objects(self, image_dir):
        detection_results = {}
        for root, _, files in os.walk(image_dir):
            for file in files:
                img_path = os.path.join(root, file)
                results = self.model_yolo(img_path)
                for r in results:
                    for box in r.boxes.data:
                        class_id = int(box[5])
                        class_name = self.model_yolo.names[class_id]
                        detection_results[class_name] = detection_results.get(class_name, 0) + 1
        return detection_results

    # 更新结果到 GUI
    def update_results(self, results):
        for i in self.result_tree.get_children():
            self.result_tree.delete(i)
        for class_name, count in results.items():
            self.result_tree.insert("", "end", values=(class_name, count))


if __name__ == "__main__":
    root = tk.Tk()
    app = TargetCountingApp(root)
    root.mainloop()
