# coding:utf-8

import numpy as np
from numpy import array
from numpy import mean
from numpy import abs
from numpy import sum
from numpy import sqrt
from numpy import log


def stde(alist):
    '''
        standard error
        标准差
    '''

    list_array = array(alist)

    stde = np.std(list_array)

    # print("The standard error is {0}".format(stde))

    return stde


def mae(actuallist, predictlist):
    '''
        mean absolute error
        平均绝对误差
    '''

    actual_array = array(actuallist)
    predict_array = array(predictlist)

    mae = mean(abs(predict_array - actual_array))

    return mae


def mape(actuallist, predictlist):
    '''
        mean absolute percentage error
        平均绝对误差率（相对百分误差绝对值的平均值）
    '''
    # actual_array = array(actuallist)
    # predict_array = array(predictlist)

    # mape = mean(abs(predict_array - actual_array) * 100 / actual_array)

    actuallist = actuallist.copy() # 避免后面的reverse产生误操作(把该变量变成类似全局变量)
    predictlist = predictlist.copy()

    if isinstance(actuallist, np.ndarray):
        actuallist = actuallist.tolist()

    if isinstance(predictlist, np.ndarray):
        predictlist = predictlist.tolist()

    actuallist.reverse()  # 最新N周数据
    predictlist.reverse() # 保证actuallist与predictlist 对等长度且右对齐（predictlist长度可能更短）

    print(actuallist, predictlist)


    if len(predictlist) == 0: # 刚开始没有历史预测值
        mape = 1
    else:
        # mape = mean([abs(m - n) * 100 / n for m, n in zip(predictlist, actuallist)])
        mape_list = [abs(m - n) * 100 / n for m, n in zip(predictlist, actuallist)]   # 分两步计算mape,主要是为了排除序列中的inf和nan值
        new_mape_list = [i for i in mape_list if np.isfinite(i)] # 剔除inf和nan值     # 因为含有inf和nan的序列计算统计值为nan
        print('new_mape_list:',new_mape_list)
        mape = mean(new_mape_list)

    if mape == 0:
        mape = 1  # 避免0作除数导致算出的权重为inf

    return mape


def ols(actuallist, predictlist):
    """
        将list转换成ndarray在训练参数作梯度下降时更快；ols比RMSE对离群值更敏感，所以RMSE更适合作商品预测的目标函数；
        因为要首先确保大多数时候预测值与真实值贴近，而不是追求在爆发点上贴近。
    """
    actualarray = np.array(actuallist)
    predictarray = np.array(predictlist)
    distance = 0.5 * sum((actualarray - predictarray) ** 2)
    return distance


def rmse(actuallist, predictlist):
    '''
        root mean squared error
        均方根误差=标准误差
    '''

    actual_array = array(actuallist)
    predict_array = array(predictlist)

    rmse = sqrt(mean((predict_array - actual_array) ** 2))

    return rmse


def rae(actuallist, predictlist):
    '''
        relative absolute error
        相对绝对误差
    '''

    actual_array = array(actuallist)
    predict_array = array(predictlist)

    rae = sum(abs(predict_array - actual_array)) / sum(abs(actual_array - mean(actual_array)))

    return rae


def rrse(actuallist, predictlist):
    '''
        root relative squared error
        相对方根误差
    '''

    actual_array = array(actuallist)
    predict_array = array(predictlist)

    rrse = sqrt(sum((predict_array - actual_array) ** 2) / sum((actual_array - mean(actual_array)) ** 2))

    return rrse


def aic(actuallist, predictlist, k):
    '''
        aic判定准则
        parameters:
            --------
            k:模型参数数量(eg.linear函数有alpha,beta参数,k为2)
    '''

    actual_array = array(actuallist)
    predict_array = array(predictlist)

    T = len(actual_array)  # 序列长度
    rss = sum((predict_array - actual_array) ** 2)  # 残差平方和

    aic = T * log(rss / T) + 2 * k

    return aic


def ds(actuallist, predictlist):
    '''
        ds 判断预测序列与实际序列的变化同步性
           返回值越接近数值1，同步性越好，预测值越接近真实值
    '''

    actual_array = array(actuallist)
    predict_array = array(predictlist)

    ds_list = []

    for i in range(len(actual_array)):
        if (actual_array[i] - actual_array[i - 1]) * (predict_array[i] - predict_array[i - 1]) > 0:
            dvalue = 1
        else:
            dvalue = 0

        ds_list.append(dvalue)

    ds = sum(ds_list) / len(ds_list)

    return ds


if __name__ == '__main__':

    act = [1, 2.3, 3.5, 3.1, 4.3]

    pre = [1.4, 2.5, 2.7, 3.9, 3.7]

    print("act标准差:", stde(act))
    print("pre标准差:", stde(pre))
    print("平均误差:", mae(act, pre))
    print("平均标准误差:", rmse(act, pre))
    print("相对误差:", rae(act, pre))
    print("相对标准误差:", rrse(act, pre))
    print("AIC判定值:", aic(act, pre, 3))
    print("DS判定值:", ds(act, pre))
    print('mape:', mape([3,2,np.nan],[1,np.nan,3]))
