"""
file: metrics
author: JDS
create date: 2021/8/2 15:18
description: 
"""
import math
from tqdm import tqdm
from onmt.optfile import *
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('-tgt', type=str, help='tgt file')
parser.add_argument('-pred', type=str, help='pred file')
parser.add_argument('-topK', type=int,default=1, help='The number of MIUs in the first k')
args = parser.parse_args()
print(args)

def MIU_score(target,predicts,topK):
    '''
    MIU定义为句子中由非汉字部分分隔的最长连续汉字序列

    S=∑1/math.pow(2,i-1)*I(Pi,C)*(|Pi|/|C|)  求和下标从1到topK

    I(Pi,C)={1， Pi is prefix of C
             0, otherwise

    :param target: 目标句子（字符串）
    :param predicts: 输入法预测的子句（列表），其中predicts列表的长度要等于topnum
    :param topK: 输入法输出的前k个子句的数目
    :return: 输出MIU得分
    '''
    tgtlen=len(target)
    score=0
    assert len(predicts)>=topK,'predicts的长度要大于等于topK'
    for i in range(topK):
        pred=predicts[i].strip()
        # if target.startswith(pred) and pred!='':  #判断第i个predict是否是target前缀
        #     score=1/math.pow(2,i)*(len(pred)/tgtlen)   #谷歌IME等计算方式
        if target==pred:
            score = 1  # 神经网络计算方式
            break
    return score


def select_topK(target,predicts,topK_list):
    scores=[]
    for topK in topK_list:
        score=eval_MIU_score(target,predicts,topK)
        scores.append(score)
    return scores

def eval_MIU_score(target_data,predict_data,topK):
    '''
    评估数据集中的MIU得分
    :param target_data:
    :param predict_data:
    :return:
    '''
    sum_score=0
    for target,predict_line in zip(target_data,predict_data):
        predicts=predict_line.split(' | ')
        score=MIU_score(target.strip(),predicts,topK)
        # if score==0:
        #     print(target.strip())
        #print(score)
        sum_score+=score
    avg_score=sum_score/len(target_data)
    return avg_score

def eval_KySS(targets,predicts):
    sum_score = 0
    cand_num=5
    for target, predict_line in zip(targets, predicts):
        target=target.strip()
        predict_line=predict_line.strip()
        topK = predict_line.split(' | ')
        ky_count=0
        statue=False    #查找是否成功
        for i in range(0,len(topK),5):
            if target in topK[i:cand_num+i]:
                ky_count+=1
                statue=True #成功查找
                break
            else:
                ky_count+=1
        if statue==False:
            score=0
        else:
            score=1/ky_count
        #print(score)
        sum_score+=score
    avg_score=sum_score/len(targets)
    return avg_score


def eval_CA(target_data,predicts):
    true_count = 0
    sum_count=0
    for target, predict_line in zip(target_data, predicts):
        tgt_words=target.split()
        predict=predict_line.split(' | ')[0]
        pred_words=predict.split()

        if len(tgt_words)>=len(pred_words):
            for i in range(len(pred_words)):
                if tgt_words[i]==pred_words[i]:
                    true_count+=1
        else:
            for i in range(len(tgt_words)):
                if tgt_words[i]==pred_words[i]:
                    true_count+=1

        sum_count+=len(tgt_words)

    return true_count/sum_count

if __name__=='__main__':
    start=time.time()
    target_file = args.tgt
    pred_file=args.pred
    topK=args.topK
    target_data = read_file(target_file)
    predict_data = read_file(pred_file)

    #测评MIU top-1
    # MIU_score=eval_MIU_score(target_data,predict_data,topK)
    # print('MIU score:{}'.format(MIU_score))
    MIU_score=select_topK(target_data,predict_data,[1,5,10])
    print('MIU score:',MIU_score)
    #
    # #测评CA
    score=eval_CA(target_data,predict_data)
    print('CA score:{}'.format(score))

    end=time.time()
    print("during time :".format(end-start))
    #测评KySS
    KySS=eval_KySS(target_data,predict_data)
    print('KySS :{}'.format(KySS))