import PyPDF2
import os

def split_pdf(input_pdf_path, output_dir, pages_per_split=10):
    # 打开原始PDF文件
    with open(input_pdf_path, "rb") as input_pdf:
        # 创建PDF阅读器对象
        pdf_reader = PyPDF2.PdfReader(input_pdf)

        # 获取PDF的总页数
        num_pages = len(pdf_reader.pages)

        # 计算拆分的组数
        num_splits = (num_pages // pages_per_split) + (1 if num_pages % pages_per_split != 0 else 0)

        # 遍历每一组，创建新的PDF文件
        for split_num in range(num_splits):
            pdf_writer = PyPDF2.PdfWriter()

            # 确定当前组的起始页和结束页
            start_page = split_num * pages_per_split
            end_page = min((split_num + 1) * pages_per_split, num_pages)

            # 将对应的页面添加到PDF写入器
            for page_num in range(start_page, end_page):
                pdf_writer.add_page(pdf_reader.pages[page_num])

            # 生成拆分后的PDF文件名
            output_pdf_path = os.path.join(output_dir, f"split_{split_num + 1}.pdf")

            # 保存拆分后的PDF
            with open(output_pdf_path, "wb") as output_pdf:
                pdf_writer.write(output_pdf)

            print(f"Pages {start_page + 1} to {end_page} have been saved to {output_pdf_path}")

if __name__ == "__main__":
    input_pdf = "output11.pdf"  # 需要拆分的PDF文件路径
    output_directory = "./split_pdfs"  # 输出目录
    pages_per_split = 10  # 每个拆分后的文件包含的页面数

    # 如果输出目录不存在，创建目录
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 调用拆分函数
    split_pdf(input_pdf, output_directory, pages_per_split)
