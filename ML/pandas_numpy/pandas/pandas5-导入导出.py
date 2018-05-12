# #pandas可以读取的格式
# read_csv
# read_excel
# read_hdf
# read_sql
# read_json
# read_msgpack
# read_html
# read_gbq
# read_stata
# read_sas
# read_clipboard
# read_pickle


import pandas as pd

# read from
data = pd.read_csv('student.csv')
print(data)

# 将数据存储为如下格式
data.to_pickle('student.pickle')