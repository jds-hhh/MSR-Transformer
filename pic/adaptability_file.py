"""
project: kNN-IME
file: adapt_file
author: JDS
create date: 2021/12/21 15:08
description: 
"""

def read_file(file):
    with open(file, 'r', encoding='UTF-8-sig') as f:
        data= f.readlines()
    return data
def write_file(file,data):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(''.join(data))

per=2000
group_num=300

cMed_src=read_file('../data/cMedQA/train-src.txt')
cMed_tgt=read_file('../data/cMedQA/train-tgt.txt')
cMed_src[-1]=cMed_src[-1]+'\n'
cMed_tgt[-1]=cMed_tgt[-1]+'\n'
cMed_src=cMed_src[:per*group_num]
cMed_tgt=cMed_tgt[:per*group_num]

mid=int(len(cMed_src)/2)

cMed_src_1=cMed_src[:mid]
cMed_tgt_1=cMed_tgt[:mid]

cMed_src_2=cMed_src[mid:]
cMed_tgt_2=cMed_tgt[mid:]

write_file('../data/cMedQA-src_60w_1.txt',cMed_src_1)
write_file('../data/cMedQA-tgt_60w_1.txt',cMed_tgt_1)

write_file('../data/cMedQA-src_60w_2.txt',cMed_src_2)
write_file('../data/cMedQA-tgt_60w_2.txt',cMed_tgt_2)



