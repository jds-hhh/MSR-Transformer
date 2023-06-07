"""
project: kNN-IME
file: lambda_pic
author: JDS
create date: 2021/10/27 9:26
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


# TP_Acc    =[42.5,44.1,46.2,49.8,53.7,59.4,62.8,63.9,64.8,64.8,64.6]
# cMedQA_Acc=[52.9,54.8,58.1,63.2,71.3,82.0,86.5,88.3,88.1,88.1,88.2]
# CAIL_Acc  =[49.5,51.7,54.0,58.5,64.3,74.8,80.4,81.8,82.4,83.0,83.1]
#
# lmbda=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

#
PD_Acc    =[90.3,90.4,90.6,90.7,90.6,90.9]
TP_Acc    =[42.5,46.1,53.7,63.2,64.9,64.5]
CAIL_Acc  =[49.5,54.0,64.3,80.4,82.4,83.1]
cMedQA_Acc=[55.9,61.5,73.9,89.4,90.5,91.0]

lmbda=[0.0,0.2,0.4,0.6,0.8,1.0]

plt.plot(lmbda,PD_Acc,'y-o',lmbda,TP_Acc,'r-s',lmbda,CAIL_Acc,'g-^',lmbda,cMedQA_Acc,'b-*')

plt.legend(["People's Daily",'Touchpal','CAIL2019','cMedQA1.0'],loc =4)
plt.grid(axis='y',linestyle='-.')
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)


plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


plt.xlabel('(a) 参数λ',fontproperties=font_prop)
plt.ylabel("Top-1准确率(%)",fontproperties=font_prop)
plt.savefig('lambda.png')
plt.show()

