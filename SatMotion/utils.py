import pandas as pd
from pathlib import Path

def merge_csv_files(input_dir, output_file, exclude_file=None):
    """
    合并指定目录下的所有 CSV 文件到一个单一的 CSV 文件中。

    参数:
    - input_dir (str or Path): 包含待合并 CSV 文件的目录路径。
    - output_file (str or Path): 合并后输出的 CSV 文件路径。
    - exclude_file (str or Path, optional): 要排除的文件路径（例如输出文件本身）。
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"输入目录 {input_dir} 不存在或不是一个目录。")

    # 获取所有 CSV 文件，排除输出文件本身（如果在同一目录下）
    csv_files = list(input_dir.glob('*.csv'))
    if exclude_file:
        exclude_file = Path(exclude_file)
        csv_files = [f for f in csv_files if f.resolve() != exclude_file.resolve()]

    if not csv_files:
        raise ValueError(f"在目录 {input_dir} 中未找到任何 CSV 文件。")

    print(f"找到 {len(csv_files)} 个 CSV 文件，将进行合并。")

    # 初始化一个空的 DataFrame 用于存储合并后的数据
    merged_df = pd.DataFrame()

    for file in csv_files:
        print(f"读取文件: {file}")
        try:
            df = pd.read_csv(file)
            merged_df = pd.concat([merged_df, df], ignore_index=True)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")

    # 保存合并后的 DataFrame 到输出文件
    try:
        merged_df.to_csv(output_file, index=False)
        print(f"合并后的文件已保存到: {output_file}")
    except Exception as e:
        print(f"保存合并文件到 {output_file} 时出错: {e}")

if __name__ == "__main__":
    # 定义输入目录和输出文件路径
    INPUT_DIR = './predict_data'          # 待合并 CSV 文件所在目录
    OUTPUT_FILE = './merged_data.csv'    # 合并后输出的 CSV 文件路径

    # 调用合并函数
    merge_csv_files(INPUT_DIR, OUTPUT_FILE, exclude_file=OUTPUT_FILE)
