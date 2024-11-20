import os
from pypdf import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from io import BytesIO

# 1. 合并多个 PDF 文件
def merge_pdfs(pdf_files, output_pdf):
    writer = PdfWriter()

    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        # 合并每个PDF文件中的每一页
        for page in reader.pages:
            writer.add_page(page)

    with open(output_pdf, "wb") as f:
        writer.write(f)

# 2. 在合并后的 PDF 中标注原文件名
def add_filename_annotations(input_pdf, output_pdf):
    # 读取现有的合并 PDF
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    # 为每一页添加文件名标注
    for page_num, page in enumerate(reader.pages):
        # 创建一个内存中的 PDF，用于绘制文件名
        packet = BytesIO()
        can = canvas.Canvas(packet)
        # 获取原文件名（假设您想在第一页上标注文件名）
        filename = f"File: {os.path.basename(input_pdf)}"
        # 在页面上添加文件名（位置可以根据需要调整）
        can.drawString(30, 30, filename)
        can.save()

        # 将 canvas 转换为 PageObject，并将其合并到当前页面
        packet.seek(0)
        new_pdf = PdfReader(packet)
        annotation_page = new_pdf.pages[0]

        # 合并文件名注释到页面
        page.merge_page(annotation_page)
        writer.add_page(page)

    # 保存修改后的 PDF
    with open(output_pdf, "wb") as f:
        writer.write(f)

# 3. 主程序逻辑
def main():
    # 定义PDF文件路径
    input_pdf_dir = "pdf"  # 替换为实际的文件夹路径
    output_pdf = "merged_output.pdf"

    # 获取目录下的所有PDF文件
    pdf_files = [os.path.join(input_pdf_dir, f) for f in os.listdir(input_pdf_dir) if f.endswith('.pdf')]

    # 合并PDF文件
    merge_pdfs(pdf_files, output_pdf)

    # 在合并后的PDF中添加文件名注释
    add_filename_annotations(output_pdf, output_pdf)

    print("PDF合并及标注完成！")

if __name__ == "__main__":
    main()
