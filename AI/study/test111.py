import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QListWidget, QPushButton, QSplitter
from PyQt5.QtCore import Qt

class PlagiarismCheckerUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("论文查重系统")
        self.setGeometry(100, 100, 800, 600)

        # 主布局
        main_layout = QVBoxLayout()

        # 水平分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧文本框，显示原始文章
        self.text_edit = QTextEdit()
        self.text_edit.setHtml("""
        这是原始文章的内容。<span style="background-color: yellow;">这部分文字的背景为黄色。</span>
        继续显示其余的文本内容。<span style="background-color: yellow;">这段文字也有黄色背景。</span>
        """)
        self.text_edit.setReadOnly(True)

        # 右侧列表，显示重复的文章
        self.repeated_list = QListWidget()
        self.repeated_list.addItem("文献1")
        self.repeated_list.addItem("文献2")
        self.repeated_list.addItem("文献3")

        # 为重复文章列表添加点击事件
        self.repeated_list.clicked.connect(self.highlight_repeated_section)

        # 按钮：触发查重功能
        self.check_button = QPushButton("开始查重")
        self.check_button.clicked.connect(self.check_plagiarism)

        # 向水平分割器添加左右控件
        splitter.addWidget(self.text_edit)
        splitter.addWidget(self.repeated_list)

        # 向主布局添加控件
        main_layout.addWidget(splitter)
        main_layout.addWidget(self.check_button)

        self.setLayout(main_layout)

    def highlight_repeated_section(self):
        # 获取选中的重复文章
        selected_item = self.repeated_list.selectedItems()
        if not selected_item:
            return
        selected_item = selected_item[0].text()

        # 根据选中的重复文章更新左侧显示
        highlighted_text = self.text_edit.toPlainText()
        if selected_item == "文献1":
            # 标红重复段落
            highlighted_text = highlighted_text.replace("这段可能和文献1重复", "<span style='background-color: red;'>这段可能和文献1重复</span>")
        elif selected_item == "文献2":
            highlighted_text = highlighted_text.replace("这段可能和文献2重复", "<span style='background-color: red;'>这段可能和文献2重复</span>")
        elif selected_item == "文献3":
            highlighted_text = highlighted_text.replace("这段可能和文献2重复", "<span style='background-color: red;'>这段可能和文献2重复</span>")

        # 设置高亮后的文本
        self.text_edit.setHtml(highlighted_text)

    def check_plagiarism(self):
        # 触发查重功能，这里可以接入TF-IDF + BERT的查重算法
        # 假设我们得到的是重复的文献1、文献2、文献3
        self.repeated_list.addItem("文献1")
        self.repeated_list.addItem("文献2")
        self.repeated_list.addItem("文献3")

        # 可以在这里执行查重算法，返回重复部分并显示
        # 这里只是做了一个简单的模拟示例

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PlagiarismCheckerUI()
    window.show()
    sys.exit(app.exec_())
