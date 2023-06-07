"""
project: kNN-IME
file: Tracing
author: JDS
create date: 2021/12/4 15:51
description: 
"""

#先将测试集进行混合

def read_file(file):
    with open(file, 'r', encoding='UTF-8-sig') as f:
        data= f.readlines()
    return data
def write_file(file,data):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(''.join(data))

per=2000
group_num=50

TP_src=read_file('../data/TP/train-src.txt')
TP_tgt=read_file('../data/TP/train-tgt.txt')
TP_src[-1]=TP_src[-1]+'\n'
TP_tgt[-1]=TP_tgt[-1]+'\n'
cMed_src=read_file('../data/cMedQA/train-src.txt')
cMed_tgt=read_file('../data/cMedQA/train-tgt.txt')
cMed_src[-1]=cMed_src[-1]+'\n'
cMed_tgt[-1]=cMed_tgt[-1]+'\n'
CAIL_src=read_file('../data/CAIL2019/train-src.txt')
CAIL_tgt=read_file('../data/CAIL2019/train-tgt.txt')
CAIL_src[-1]=CAIL_src[-1]+'\n'
CAIL_tgt[-1]=CAIL_tgt[-1]+'\n'
mix_src=[]
mix_tgt=[]

for i in range(0,2):
    mix_src.extend(TP_src[int(i*per*group_num):int((i+1)*per*group_num)])
    mix_tgt.extend(TP_tgt[int(i*per*group_num):int((i+1)*per*group_num)])

    mix_src.extend(cMed_src[int(i * per*group_num):int((i + 1) * per*group_num)])
    mix_tgt.extend(cMed_tgt[int(i * per*group_num):int((i + 1) * per*group_num)])

    mix_src.extend(CAIL_src[int(i * per*group_num):int((i + 1) * per*group_num)])
    mix_tgt.extend(CAIL_tgt[int(i * per*group_num):int((i + 1) * per*group_num)])


print(len(mix_src))
write_file('../data/TP_cMedQA_CAIL-src.txt',mix_src)
write_file('../data/TP_cMedQA_CAIL-tgt.txt',mix_tgt)

mid=int(len(mix_src)/2)
write_file('../data/TP_cMedQA_CAIL-src-1.txt',mix_src[0:mid])
write_file('../data/TP_cMedQA_CAIL-tgt-1.txt',mix_tgt[0:mid])

write_file('../data/TP_cMedQA_CAIL-src-2.txt',mix_src[mid:])
write_file('../data/TP_cMedQA_CAIL-tgt-2.txt',mix_tgt[mid:])
# write_file('../data/cMedQA-src.txt',mix_src)
# write_file('../data/cMedQA-tgt.txt',mix_tgt)



# # mix_src.extend(PD_src[0:group_num])
# # mix_tgt.extend(PD_tgt[0:group_num])
# mix_src.extend(TP_src[0:group_num])
# mix_tgt.extend(TP_tgt[0:group_num])
# mix_src.extend(cMed_src[0:group_num])
# mix_tgt.extend(cMed_tgt[0:group_num])
# mix_src.extend(CAIL_src[0:group_num])
# mix_tgt.extend(CAIL_tgt[0:group_num])
#
#
#
#
# # mix_src.extend(PD_src[group_num:])
# # mix_tgt.extend(PD_tgt[group_num:])
# mix_src.extend(TP_src[group_num:])
# mix_tgt.extend(TP_tgt[group_num:])
# mix_src.extend(cMed_src[group_num:])
# mix_tgt.extend(cMed_tgt[group_num:])
# mix_src.extend(CAIL_src[group_num:])
# mix_tgt.extend(CAIL_tgt[group_num:])
#
# print(len(mix_src),len(mix_tgt))
# write_file('../data/mix_data_src.txt',mix_src)
# write_file('../data/mix_data_tgt.txt',mix_tgt)