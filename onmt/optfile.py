"""
project: NLP
file: optfile
author: JDS
create date: 2021/8/2 19:13
description: 
"""

def read_file(filename):
    print('读取{}数据'.format(filename))
    with open(filename,'r',encoding='UTF-8-sig') as f:
        data=f.readlines()
    print('{}文本长度:{}'.format(filename,len(data)))
    return data

def write_file(filename,data):
    print('写入{}数据，数据长度:{}'.format(filename,len(data)))
    with open(filename,'w',encoding='utf-8') as f:
        f.write('\n'.join(data))

def countChar(name,data):    #对数据字符进行计数,同时统计平均MIU的平均长度
    count=0
    for d in data:
        count+=len(d)
    avg_len=count/len(data)
    print('{}一共有{}中文字符，一共有{}个MIU,MIU平均长度为{}'.format(name,count,len(data),avg_len))