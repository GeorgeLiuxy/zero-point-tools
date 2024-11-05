import os
import csv

def merge_csv_from_folder(folder_path, output_file):
    # 获取文件夹中所有的CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 打开输出CSV文件
    with open(output_file, mode='w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)

        # 用来标记是否写入表头
        header_written = False

        # 遍历每一个CSV文件
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            with open(file_path, mode='r', newline='', encoding='utf-8') as in_file:
                reader = csv.reader(in_file)

                # 读取表头（第一行）
                header = next(reader)

                # 如果是第一个文件，写入表头
                if not header_written:
                    writer.writerow(header)
                    header_written = True

                # 将文件中的所有行写入输出CSV文件
                for row in reader:
                    writer.writerow(row)

    print(f"文件夹中的所有CSV文件已合并并保存为: {output_file}")

# 调用函数
folder_path = '/Users/george/Downloads/导出数据'  # 文件夹路径，替换为你自己的文件夹路径
output_file = 'merged_output.csv'  # 合并后的输出文件路径

merge_csv_from_folder(folder_path, output_file)
