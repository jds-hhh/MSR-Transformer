"""
project: kNN-IME
file: update_MIU_num_pic
author: JDS
create date: 2021/10/27 10:01
description: 
"""

import matplotlib.pyplot as plt
import random
import numpy as np
# 设置中文宋体字体
from matplotlib.font_manager import FontProperties

font_path = '../SimSun.ttf'  # 请将字体文件放在当前目录下或者指定字体文件的路径
font_prop = FontProperties(fname=font_path, size=18)
# 设置英文 Times New Roman 字体
plt.rcParams['font.family'] = ['Times New Roman']

# Touchpal=[42.5,48.2,53.5,55.7,58.0,61.1,62.9,64.8]
# cMedQA  =[52.9,66.7,73.1,75.8,79.5,81.8,83.3,88.1]
# CAIL2019=[50.0,71.8,75.0,77.3,77.9,81.2,82.2,82.4]
# data_size=['0','1','3','5','10','20','30','all']

Touchpal=[42.5,59.9,62.4,63.5,64.1,64.9]
CAIL2019=[49.5,78.9,80.9,81.4,82.3,82.4]
cMedQA  =[55.9,86.7,88.6,89.6,90.4,90.5]
data_size=['0.0','0.2','0.4','0.6','0.8','1.0']

plt.plot(data_size,Touchpal,'r-s',data_size,CAIL2019,'g-^',data_size,cMedQA,'b-*')

plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

plt.legend(['Touchpal','CAIL2019','cMedQA1.0'],loc =4)
plt.grid(axis='y',linestyle='-.')
#plt.title('Datastore size: Touchpal=68.8K, cMedQA1.0=1.5M,CAIL2019=38.0K')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlabel('(b) 训练集比例',fontproperties=font_prop)
plt.ylabel("Top-1准确率(%)",fontproperties=font_prop)
plt.savefig('MIU_update.png')
plt.show()




