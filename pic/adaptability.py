"""
project: kNN-IME
file: adaptability
author: JDS
create date: 2021/12/21 14:29
description: 
"""
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import math

# 设置中文宋体字体
from matplotlib.font_manager import FontProperties

font_path = '../SimSun.ttf'  # 请将字体文件放在当前目录下或者指定字体文件的路径
font_prop = FontProperties(fname=font_path, size=18)
# 设置英文 Times New Roman 字体
plt.rcParams['font.family'] = ['Times New Roman']
def read_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        data= f.readlines()
    return data

def add_space(data):
    newdata=[' '.join(d) for d in data]
    return newdata

per=2000
tgts=read_file('../data/cMedQA_60w-tgt.txt')
preds_trace=read_file('../data/cMedQA_60w_pred_trace_1.txt')
preds_trace_2=read_file('../data/cMedQA_60w_pred_trace_2.txt')
preds_trace.extend(preds_trace_2)
preds_base=read_file('../data/cMedQA_60w_pred_base.txt')
preds_omwa=read_file('../data/cMedQA_60w_OMWA.txt')
preds_omwa=add_space(preds_omwa)

x_base=[]
y_base=[]
x_trace=[]
y_trace=[]
x_omwa=[]
y_omwa=[]

score_trace=0
score_base=0
score_omwa=0
for idx,(tgt,pred_trace,pred_omwa,preds_base) in tqdm(enumerate(zip(tgts,preds_trace,preds_omwa,preds_base)),desc='scoring'):
    pred_trace = pred_trace.strip()
    preds_base = preds_base.strip()
    pred_omwa=pred_omwa.strip()
    tgt = tgt.strip()
    if pred_trace == tgt:
        score_trace += 1
    if preds_base==tgt:
        score_base+=1
    if pred_omwa==tgt:
        score_omwa+=1
    if (idx + 1) % per == 0:
        x_base.append(int((idx + 1) / per))
        y_base.append(score_base / per)
        score_base = 0  # 分数清0

        x_trace.append(int((idx + 1) / per))
        y_trace.append(score_trace / per)
        score_trace = 0  # 分数清0

        x_omwa.append(int((idx + 1) / per))
        y_omwa.append(score_omwa / per)
        score_omwa = 0  # 分数清0


plt.figure(dpi=300,figsize=(12,6))
plt.plot(x_base,y_base,linestyle='-',color='b')
plt.plot(x_omwa,y_omwa,linestyle='-',color='y')
plt.plot(x_trace,y_trace,'-',color = 'r')
plt.grid(axis='y',linestyle='-.')
plt.grid(axis='x',linestyle='-.')

plt.xlim((0,300))
plt.ylim((0.3, 0.95))


#x_major_locator=MultipleLocator(50)
#ax为两条坐标轴的实例
#ax = plt.gca()
#ax.xaxis.set_major_locator(x_major_locator)
#ax.set_xticklabels(['0','0','50(Touchpal)', '100(cMedQA)','150(CAIL)',])


plt.tick_params(labelsize=14)
plt.legend(['PD Transformer','On-OMWA',"MSR Transformer"],loc =2)
plt.ylabel('每组的Top-1准确率(%)得分',fontproperties=font_prop)
plt.xlabel('MIU组', fontproperties=font_prop)

plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.1)


plt.savefig('adaptability.png')
plt.show()

