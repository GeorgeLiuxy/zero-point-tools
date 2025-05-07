import pandas as pd
import numpy as np
import pickle as pk
import datetime as dt

print('read csv files')
print('Note that this study takes the 2019 data')
trn = pd.read_csv('C:/Users/LENOVO/PycharmProjects/Project/WaDi.A2_19 Nov 2019/WADI.A2_19 Nov 2019/WADI_14days_new.csv')
tst = pd.read_csv('C:/Users/LENOVO/PycharmProjects/Project/WaDi.A2_19 Nov 2019/WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv',skiprows=1)

# print('shorten column labels and separate labels')
# # shorten column labels
# cols = trn.columns.to_numpy()
# target_str = '\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\'
# for i in range(len(cols)):
#     if target_str in cols[i]:
#         cols[i] = cols[i][len(target_str):]
# trn.columns = cols
# lab_tst = tst[tst.columns[-1]].to_numpy()
print('shorten column labels and separate labels')

# 清洗训练集的列名
target_str = '\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\'
trn.columns = [c[len(target_str):] if target_str in c else c for c in trn.columns]

# 分离标签：tst 最后一列为标签
lab_tst = tst.iloc[:, -1].to_numpy()
print("Unique labels in lab_tst:", set(lab_tst))
assert len(set(lab_tst)) == 2

# 移除标签列
tst = tst.iloc[:, :-1]

# 对齐测试集特征的列名（使用训练集处理后的列名）
tst.columns = trn.columns

###########################################
lab_tst = tst[tst.columns[-1]].to_numpy()

# 输出唯一值，检查是否包含多余的标签
print("Unique labels in lab_tst:", set(lab_tst))


assert len(set(lab_tst)) == 2
'''assert len(set(lab_tst)) == 2'''
#############################################
# 读取 tst 数据，查看列数
print("tst 数据的列数:", len(tst.columns))
print(tst.columns)
# 确保 cols 数组长度与 tst 数据的列数一致
cols = cols[:len(tst.columns)]  # 截取与 tst 列数一致的 cols
print("新的列名数量:", len(cols))

# 应用新的列名
tst.columns = cols

'''tst = tst.drop(columns = [tst.columns[-1]])
tst.columns = cols'''
###################################################
print('drop columns and rows')
# drop Row, Date, Time
trn = trn[cols[3:]]
tst = tst[cols[3:]]
cols = cols[3:]

# drop columns that have excessive NaNs
drop_cols = cols[np.isnan(trn.to_numpy()).sum(axis=0) > len(trn) // 2]
tst = tst.drop(columns=drop_cols)
trn = trn.drop(columns=drop_cols)

# convert to numpy array
print('convert to numpy array')
trn_np = trn.to_numpy()
tst_np = tst.to_numpy()
cols = trn.columns.to_numpy()

# fill NAs
print('fill NAs for trn')
nanlist = np.isnan(trn_np).sum(axis=0)
print(nanlist)
for j, nancnt in enumerate(nanlist):
    if nancnt > 0:
        for i in range(len(trn_np)):
            if np.isnan(trn_np[i, j]):
                trn_np[i, j] = trn_np[i - 1, j]
                nancnt -= 1
                if nancnt == 0:
                    break
assert np.isnan(trn_np).sum() == 0 and np.isnan(tst_np).sum() == 0

print('save to pickle file')
with open('WADI_new.pk', 'wb') as file:
    pk.dump({'x_trn': trn_np, 'x_tst': tst_np, 'lab_tst': lab_tst, 'cols': cols}, file)

print('done, final x_trn, x_tst, lab_tst shape: ', trn_np.shape, tst_np.shape, lab_tst.shape)