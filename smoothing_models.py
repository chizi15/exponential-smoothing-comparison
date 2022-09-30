# coding:utf-8

from __future__ import division
import numpy as np
from sys import exit
from scipy import optimize
import math

"""
    指数平滑类模型：包括单参数模型simple，双参数加法模型linear，双参数乘法模型double_mul，三参数加法模型additive，三参数乘法模型multiplicative，
    六参数加法多重季节性日序列模型multiseasonal_add_day，五参数加法多重季节性周序列模型multiseasonal_add_week，
    六参数乘法多重季节性日序列模型multiseasonal_mul_day，五参数乘法多重季节性周序列模型multiseasonal_mul_week。
    对于无季节性模型simple、linear、double_mul，min(len(Y)) >= 2，
    对于单一季节性模型additive和multiplicative，min(len(Y)) >= m，
    对于多重季节性模型multiseasonal_add_day、multiseasonal_add_week、multiseasonal_mul_day、multiseasonal_mul_week，min(len(Y)) >= max(m).
    
    目标函数可采用能量化两条序列偏离程度且能做梯度下降的RMSE或OLS；在所有平滑类模型
    
    
    计算各模型中预测序列与真实序列在对应日期上的均方根误差。
"""


def ols(actuallist, predictlist):
    """
        将list转换成ndarray在训练参数时更快；ols比RMSE对离群值更敏感，所以RMSE更适合作商品预测的目标函数；
        因为要首先确保大多数时候预测值与真实值贴近，而不是追求在爆发点上贴近。
    """
    actualarray = np.array(actuallist)
    predictarray = np.array(predictlist)
    distance = 0.5 * sum((actualarray - predictarray) ** 2)
    return distance


def rmse(actuallist, predictlist):
    """
        root mean squared error
        均方根误差=标准误差
    """
    actual_array = np.array(actuallist)
    predict_array = np.array(predictlist)
    rmse_distance = np.sqrt(np.mean((predict_array - actual_array) ** 2))
    return rmse_distance


def aic(actuallist, predictlist, k):
    """
        aic判定准则
        parameters:
            --------
            k:模型参数数量(eg.linear函数有alpha,beta参数,k为2)
    """
    actual_array = np.array(actuallist)
    predict_array = np.array(predictlist)

    n = len(actual_array)  # 序列长度
    rss = sum((predict_array - actual_array) ** 2)  # 残差平方和
    aic = n * np.log(rss / n) + 2 * k

    return aic


def rmse_loss_func(params, *args):
    """
        构造平滑类模型的rmse目标函数，params是自变量，return返回的是因变量，args是需要外部输入的超参数，优化算法作用于该目标函数。

        计算均方根误差
        parameters:
            --------
            params:包含自变量alpha,beta,gamma,gamma_year,gamma_quarter,gamma_month,gamma_week的初始值元组。
            args:包含原始计算序列Y(args[0])、模型type(args[1])、周期m,m_year, m_quarter, m_month, m_week(args[2])的元组。

        return:
            --------
            rmse:原始序列与预测序列间的均方根误差
    """

    Y = args[0]
    model = args[1]

    if model == 'simple':
        """
            单参数模型，适用于预测无趋势无季节性序列，其预测结果为一水平直线。
            alpha是自变量，rmse(Y[1:], y[1:-1])是目标函数。
            在Rmse中alpha的最高次数是一次，所以此时Rmse为凸函数，局部优化和全局优化算法的解相同。
        """
        alpha = params[0]  # 将alpha定义为自变量
        '''给截距序列a及预测序列y赋初值'''
        a = [sum(Y[:]) / float(len(Y))]
        y = [a[0]]

        # 构造截距序列a及预测序列y
        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * a[i])  # 此处的a与Y均是预测序列
            y.append(a[i + 1])

    elif model == 'linear':
        '''
            alpha、beta是自变量，rmse(Y[1:], y[1:-1])是目标函数；在Rmse中alpha、beta的最高次数是一次，所以此时Rmse为凸函数，局部优化和全局优化算法的解相同。
        '''
        alpha, beta = params
        # 给截距序列a、斜率序列b、预测序列y赋初值
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        y = [a[0] + b[0]]

        # 构造截距序列a、斜率序列b、预测序列y
        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            y.append(a[i + 1] + b[i + 1])

    elif model == 'double_mul':
        '''
            alpha、beta是自变量，rmse(Y[1:], y[1:-1])是目标函数；在Rmse中alpha、beta的最高次数是一次，所以此时Rmse为凸函数，局部优化和全局优化算法的解相同。
        '''
        alpha, beta = params
        # 给截距序列a、斜率序列b、预测序列y赋初值
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        y = [a[0] * b[0]]

        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * (a[i] * b[i]))
            b.append(beta * (a[i + 1] / a[i]) + (1 - beta) * b[i])
            y.append(a[i + 1] * b[i + 1])

    elif model == 'additive':

        alpha, beta, gamma = params
        m = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s = (np.array(Y[:len(Y) // m * m]).reshape(-1, m).mean(axis=0) - a[0]).tolist()
        y = [(a[0] + b[0]) + np.mean(s)]

        for i in range(len(Y)):
            a.append(alpha * (Y[i] - s[-m]) + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s.append(gamma * (Y[i] - a[i + 1]) + (1 - gamma) * s[-m])
            y.append(a[i + 1] + b[i + 1] + s[1 - m])

    elif model == 'multiplicative':

        alpha, beta, gamma = params
        m = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s = (np.array(Y[:len(Y) // m * m]).reshape(-1, m).mean(axis=0) / a[0]).tolist()
        y = [(a[0] + b[0]) + np.mean(s)]

        for i in range(len(Y)):
            a.append(alpha * (Y[i] / s[-m]) + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s.append(gamma * (Y[i] / a[i + 1]) + (1 - gamma) * s[-m])
            y.append((a[i + 1] + b[i + 1]) * s[1 - m])

    elif model == 'multiseasonal_add_day':

        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = params
        m_year, m_quarter, m_month, m_week = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
        s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) - a[0]).tolist()
        y = [(a[0] + b[0]) + np.mean(s_year) + np.mean(s_quarter) + np.mean(s_month) + np.mean(s_week)]

        for i in range(len(Y)):
            a.append(
                alpha * (Y[i] - (s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month] + s_week[-m_week]))
                + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s_year.append(
                gamma_year * (Y[i] - (a[i + 1] + s_quarter[-m_quarter] + s_month[-m_month] + s_week[-m_week]))
                + (1 - gamma_year) * s_year[-m_year])
            s_quarter.append(
                gamma_quarter * (Y[i] - (a[i + 1] + s_year[-m_year] + s_month[-m_month] + s_week[-m_week]))
                + (1 - gamma_quarter) * s_quarter[-m_quarter])
            s_month.append(
                gamma_month * (Y[i] - (a[i + 1] + s_year[-m_year] + s_quarter[-m_quarter] + s_week[-m_week]))
                + (1 - gamma_month) * s_month[-m_month])
            s_week.append(
                gamma_week * (Y[i] - (a[i + 1] + s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month]))
                + (1 - gamma_week) * s_week[-m_week])
            y.append((a[i + 1] + b[i + 1]) + s_year[1 - m_year] + s_quarter[1 - m_quarter] + s_month[1 - m_month] +
                     s_week[1 - m_week])

    elif model == 'multiseasonal_add_week':

        alpha, beta, gamma_year, gamma_quarter, gamma_month = params
        m_year, m_quarter, m_month = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
        y = [(a[0] + b[0]) + np.mean(s_year) + np.mean(s_quarter) + np.mean(s_month)]

        for i in range(len(Y)):
            a.append(
                alpha * (Y[i] - (s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month]))
                + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s_year.append(
                gamma_year * (Y[i] - (a[i + 1] + s_quarter[-m_quarter] + s_month[-m_month]))
                + (1 - gamma_year) * s_year[-m_year])
            s_quarter.append(
                gamma_quarter * (Y[i] - (a[i + 1] + s_year[-m_year] + s_month[-m_month]))
                + (1 - gamma_quarter) * s_quarter[-m_quarter])
            s_month.append(
                gamma_month * (Y[i] - (a[i + 1] + s_year[-m_year] + s_quarter[-m_quarter]))
                + (1 - gamma_month) * s_month[-m_month])
            y.append((a[i + 1] + b[i + 1]) + s_year[1 - m_year] + s_quarter[1 - m_quarter] + s_month[1 - m_month])

    elif model == 'multiseasonal_mul_day':

        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = params
        m_year, m_quarter, m_month, m_week = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
        s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) / a[0]).tolist()
        y = [(a[0] + b[0]) * np.mean(s_year) * np.mean(s_quarter) * np.mean(s_month) * np.mean(s_week)]

        for i in range(len(Y)):
            a.append(
                alpha * (Y[i] / (s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month] * s_week[-m_week]))
                + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s_year.append(
                gamma_year * (Y[i] / (a[i + 1] * s_quarter[-m_quarter] * s_month[-m_month] * s_week[-m_week]))
                + (1 - gamma_year) * s_year[-m_year])
            s_quarter.append(
                gamma_quarter * (Y[i] / (a[i + 1] * s_year[-m_year] * s_month[-m_month] * s_week[-m_week]))
                + (1 - gamma_quarter) * s_quarter[-m_quarter])
            s_month.append(
                gamma_month * (Y[i] / (a[i + 1] * s_year[-m_year] * s_quarter[-m_quarter] * s_week[-m_week]))
                + (1 - gamma_month) * s_month[-m_month])
            s_week.append(
                gamma_week * (Y[i] / (a[i + 1] * s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month]))
                + (1 - gamma_week) * s_week[-m_week])
            y.append((a[i + 1] + b[i + 1]) * s_year[1 - m_year] * s_quarter[1 - m_quarter] * s_month[1 - m_month] *
                     s_week[1 - m_week])

    elif model == 'multiseasonal_mul_week':

        alpha, beta, gamma_year, gamma_quarter, gamma_month = params
        m_year, m_quarter, m_month = args[2]
        a = [sum(Y) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
        y = [(a[0] + b[0]) * np.nanmean(s_year) * np.nanmean(s_quarter) * np.nanmean(s_month)]  # y[0]应代表真实序列Y的平均信息，但不会用到。

        for i in range(len(Y)):
            a.append(
                alpha * (Y[i] / (s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month]))
                + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s_year.append(
                gamma_year * (Y[i] / (a[i + 1] * s_quarter[-m_quarter] * s_month[-m_month]))
                + (1 - gamma_year) * s_year[-m_year])
            s_quarter.append(
                gamma_quarter * (Y[i] / (a[i + 1] * s_year[-m_year] * s_month[-m_month]))
                + (1 - gamma_quarter) * s_quarter[-m_quarter])
            s_month.append(
                gamma_month * (Y[i] / (a[i + 1] * s_year[-m_year] * s_quarter[-m_quarter]))
                + (1 - gamma_month) * s_month[-m_month])
            y.append((a[i + 1] + b[i + 1]) * s_year[1 - m_year] * s_quarter[1 - m_quarter] * s_month[1 - m_month])

    else:
        exit('Type must be in one of smoothing models')
    '''
        各分项的初始值采用全体真实值的平均信息后，预测值y[0]也代表平均信息，与Y[0]不再具备对应关系；而预测值y[1]是由真实值Y[0]等向后推出，
        则y[1]应与Y[1]匹配；即y[n+1]是由Y[n]算出，则y[n+1]与Y[n+1]匹配。又因为y具有人为赋初值y[0]，len(y)==len(Y+1)，Y[-1]产生预测值y[-1],
        则没有真实值与y[-1]匹配，则最后一个真实值Y[-1]与倒数第二个预测值y[-2]匹配。综上，匹配关系为：Y[1:], y[1:-1]。
    '''
    rmse_distance = rmse(Y[1:], y[1:-1])  # 构造目标函数Rmse

    return rmse_distance


def ols_loss_func(params, *args):
    """
        构造平滑类模型的最小二乘目标函数，params是自变量，优化算法作用于该目标函数。

        parameters:
            --------
            params:包含自变量alpha,beta,gamma,gamma_year,gamma_quarter,gamma_month,gamma_week的初始值元组。
            args:包含原始计算序列Y(args[0])、模型type(args[1])、周期m,m_year, m_quarter, m_month, m_week(args[2])的元组。

        return:
            --------
            ols_distance:原始序列与预测序列各对应元素间距离之和。
    """

    Y = args[0]
    model = args[1]

    if model == 'simple':
        """
            单参数模型，适用于预测无趋势无季节性序列，其预测结果为一水平直线。
            alpha是自变量，rmse(Y[1:], y[1:-1])是目标函数。
            在Rmse中alpha的最高次数是一次，所以此时Rmse为凸函数，局部优化和全局优化算法的解相同。
        """
        alpha = params[0]  # 将alpha定义为自变量
        '''给截距序列a及预测序列y赋初值'''
        a = [sum(Y[:]) / float(len(Y))]
        y = [a[0]]

        # 构造截距序列a及预测序列y
        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * a[i])  # 此处的a与Y均是预测序列
            y.append(a[i + 1])

    elif model == 'linear':
        '''
            alpha、beta是自变量，rmse(Y[1:], y[1:-1])是目标函数；在Rmse中alpha、beta的最高次数是一次，所以此时Rmse为凸函数，局部优化和全局优化算法的解相同。
        '''
        alpha, beta = params
        # 给截距序列a、斜率序列b、预测序列y赋初值
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        y = [a[0] + b[0]]

        # 构造截距序列a、斜率序列b、预测序列y
        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            y.append(a[i + 1] + b[i + 1])

    elif model == 'double_mul':
        '''
            alpha、beta是自变量，ols(Y[1:], y[1:-1])是目标函数；在ols中alpha、beta的最高次数是一次，所以此时ols为凸函数，局部优化和全局优化算法的解相同。
        '''
        alpha, beta = params
        # 给截距序列a、斜率序列b、预测序列y赋初值
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        y = [a[0] * b[0]]

        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * (a[i] * b[i]))
            b.append(beta * (a[i + 1] / a[i]) + (1 - beta) * b[i])
            y.append(a[i + 1] * b[i + 1])

    elif model == 'additive':

        alpha, beta, gamma = params
        m = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s = (np.array(Y[:len(Y) // m * m]).reshape(-1, m).mean(axis=0) - a[0]).tolist()
        y = [(a[0] + b[0]) + np.mean(s)]

        for i in range(len(Y)):
            a.append(alpha * (Y[i] - s[-m]) + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s.append(gamma * (Y[i] - a[i + 1]) + (1 - gamma) * s[-m])
            y.append(a[i + 1] + b[i + 1] + s[1 - m])

    elif model == 'multiplicative':

        alpha, beta, gamma = params
        m = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s = (np.array(Y[:len(Y) // m * m]).reshape(-1, m).mean(axis=0) / a[0]).tolist()
        y = [(a[0] + b[0]) + np.mean(s)]

        for i in range(len(Y)):
            a.append(alpha * (Y[i] / s[-m]) + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s.append(gamma * (Y[i] / a[i + 1]) + (1 - gamma) * s[-m])
            y.append((a[i + 1] + b[i + 1]) * s[1 - m])

    elif model == 'multiseasonal_add_day':

        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = params
        m_year, m_quarter, m_month, m_week = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
        s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) - a[0]).tolist()
        y = [(a[0] + b[0]) + np.mean(s_year) + np.mean(s_quarter) + np.mean(s_month) + np.mean(s_week)]

        for i in range(len(Y)):
            a.append(
                alpha * (Y[i] - (s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month] + s_week[-m_week]))
                + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s_year.append(
                gamma_year * (Y[i] - (a[i + 1] + s_quarter[-m_quarter] + s_month[-m_month] + s_week[-m_week]))
                + (1 - gamma_year) * s_year[-m_year])
            s_quarter.append(
                gamma_quarter * (Y[i] - (a[i + 1] + s_year[-m_year] + s_month[-m_month] + s_week[-m_week]))
                + (1 - gamma_quarter) * s_quarter[-m_quarter])
            s_month.append(
                gamma_month * (Y[i] - (a[i + 1] + s_year[-m_year] + s_quarter[-m_quarter] + s_week[-m_week]))
                + (1 - gamma_month) * s_month[-m_month])
            s_week.append(
                gamma_week * (Y[i] - (a[i + 1] + s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month]))
                + (1 - gamma_week) * s_week[-m_week])
            y.append((a[i + 1] + b[i + 1]) + s_year[1 - m_year] + s_quarter[1 - m_quarter] + s_month[1 - m_month] +
                     s_week[1 - m_week])

    elif model == 'multiseasonal_add_week':

        alpha, beta, gamma_year, gamma_quarter, gamma_month = params
        m_year, m_quarter, m_month = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
        y = [(a[0] + b[0]) + np.mean(s_year) + np.mean(s_quarter) + np.mean(s_month)]

        for i in range(len(Y)):
            a.append(
                alpha * (Y[i] - (s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month]))
                + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s_year.append(
                gamma_year * (Y[i] - (a[i + 1] + s_quarter[-m_quarter] + s_month[-m_month]))
                + (1 - gamma_year) * s_year[-m_year])
            s_quarter.append(
                gamma_quarter * (Y[i] - (a[i + 1] + s_year[-m_year] + s_month[-m_month]))
                + (1 - gamma_quarter) * s_quarter[-m_quarter])
            s_month.append(
                gamma_month * (Y[i] - (a[i + 1] + s_year[-m_year] + s_quarter[-m_quarter]))
                + (1 - gamma_month) * s_month[-m_month])
            y.append((a[i + 1] + b[i + 1]) + s_year[1 - m_year] + s_quarter[1 - m_quarter] + s_month[1 - m_month])

    elif model == 'multiseasonal_mul_day':

        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = params
        m_year, m_quarter, m_month, m_week = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
        s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) / a[0]).tolist()
        y = [(a[0] + b[0]) * np.mean(s_year) * np.mean(s_quarter) * np.mean(s_month) * np.mean(s_week)]

        for i in range(len(Y)):
            a.append(
                alpha * (Y[i] / (s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month] * s_week[-m_week]))
                + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s_year.append(
                gamma_year * (Y[i] / (a[i + 1] * s_quarter[-m_quarter] * s_month[-m_month] * s_week[-m_week]))
                + (1 - gamma_year) * s_year[-m_year])
            s_quarter.append(
                gamma_quarter * (Y[i] / (a[i + 1] * s_year[-m_year] * s_month[-m_month] * s_week[-m_week]))
                + (1 - gamma_quarter) * s_quarter[-m_quarter])
            s_month.append(
                gamma_month * (Y[i] / (a[i + 1] * s_year[-m_year] * s_quarter[-m_quarter] * s_week[-m_week]))
                + (1 - gamma_month) * s_month[-m_month])
            s_week.append(
                gamma_week * (Y[i] / (a[i + 1] * s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month]))
                + (1 - gamma_week) * s_week[-m_week])
            y.append((a[i + 1] + b[i + 1]) * s_year[1 - m_year] * s_quarter[1 - m_quarter] * s_month[1 - m_month] *
                     s_week[1 - m_week])

    elif model == 'multiseasonal_mul_week':

        alpha, beta, gamma_year, gamma_quarter, gamma_month = params
        m_year, m_quarter, m_month = args[2]
        a = [sum(Y) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
        y = [(a[0] + b[0]) * np.nanmean(s_year) * np.nanmean(s_quarter) * np.nanmean(
            s_month)]  # y[0]应代表真实序列Y的平均信息，但不会用到。

        for i in range(len(Y)):
            a.append(
                alpha * (Y[i] / (s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month]))
                + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s_year.append(
                gamma_year * (Y[i] / (a[i + 1] * s_quarter[-m_quarter] * s_month[-m_month]))
                + (1 - gamma_year) * s_year[-m_year])
            s_quarter.append(
                gamma_quarter * (Y[i] / (a[i + 1] * s_year[-m_year] * s_month[-m_month]))
                + (1 - gamma_quarter) * s_quarter[-m_quarter])
            s_month.append(
                gamma_month * (Y[i] / (a[i + 1] * s_year[-m_year] * s_quarter[-m_quarter]))
                + (1 - gamma_month) * s_month[-m_month])
            y.append((a[i + 1] + b[i + 1]) * s_year[1 - m_year] * s_quarter[1 - m_quarter] * s_month[1 - m_month])

    else:
        exit('Type must be in one of smoothing models')

    '''
        各分项的初始值采用全体真实值的平均信息后，预测值y[0]也代表平均信息，与Y[0]不再具备对应关系；而预测值y[1]是由真实值Y[0]等向后推出，
        则y[1]应与Y[1]匹配；即y[n+1]是由Y[n]算出，则y[n+1]与Y[n+1]匹配。又因为y具有人为赋初值y[0]，len(y)==len(Y+1)，Y[-1]产生预测值y[-1],
        则没有真实值与y[-1]匹配，则最后一个真实值Y[-1]与倒数第二个预测值y[-2]匹配。综上，匹配关系为：Y[1:], y[1:-1]。
    '''
    ols_distance = ols(Y[1:], y[1:-1])

    return ols_distance


def simple(x, fc, alpha=None):
    """
        单参数指数平滑预测模型
        parameters:
            --------
            x:时间序列数据列表
            fc:预测期数
            alpha:alpha参数,初始值默认为空

        return:
            --------
            predict:预测值
            alpha:最优alpha值
            rmse
            aic
            std_residual:真实序列与预测序列残差的标准差
    """

    Y = x[:]

    if alpha is None:
        initial_values = np.array([0.2])
        boundaries = [(1/2*1e-1, 1-1/2*1e-1)]
        model = 'simple'
        para_rmse_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model), method='SLSQP', jac='2-point',
                                            bounds=boundaries)
        # para_ols_slsqp = optimize.minimize(ols_loss_func, initial_values, args=(Y, model), method='SLSQP', jac='2-point',
        #                                    bounds=boundaries)
        # para_differential_evolution = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(
        #     Y, model), updating='deferred', workers=-1)
        print('rmse_loss_func', para_rmse_slsqp.x)
        # print('ols_loss_func', para_ols_slsqp.x)
        # print('para_differential_evolution', para_differential_evolution.x)
        alpha = float(para_rmse_slsqp.x)

    a = [sum(Y[:]) / float(len(Y))]
    y = [a[0]]

    for i in range(len(Y) + fc):
        if i == len(Y):
            Y.append(a[-1])

        a.append(alpha * Y[i] + (1 - alpha) * a[i])
        y.append(a[i + 1])

    rmse_index = rmse(Y[1:-fc], y[1:-fc-1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 2)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])  # 残差序列的标准差

    return {'predict': Y[-fc:], 'alpha': alpha, 'rmse': rmse_index, 'aic': aic_index, 'std_residual': std_residual, 'fittedvalues': y[1:-fc]}


def linear(x, fc, alpha=None, beta=None):
    """
        双参数指数平滑预测模型
        parameters:
            --------
            x:时间序列数据列表
            fc:预测期数
            alpha:alpha参数,初始值默认为空
            beta:beta参数,初始默认值为空

        return:
            --------
            pred:预测值
            alpha:最优alpha值
            beta:最优beta值
            rmse
            aic
            std_residual:残差标准差
    """

    Y = x[:]

    if alpha is None or beta is None:

        initial_values = np.array([0.2]*2)
        boundaries = [(1/2*1e-1, 1-1/2*1e-1)]*2
        model = 'linear'

        # para_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model), method='SLSQP', jac='2-point',
        #                                bounds=boundaries)
        # alpha, beta = para_slsqp.x

        para_rmse_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model), method='SLSQP', jac='2-point',
                                            bounds=boundaries)
        # para_ols_slsqp = optimize.minimize(ols_loss_func, initial_values, args=(Y, model), method='SLSQP', jac='2-point',
        #                                    bounds=boundaries)
        # para_differential_evolution = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(
        #     Y, model), updating='deferred', workers=-1)
        print('rmse_loss_func', para_rmse_slsqp.x)
        # print('ols_loss_func', para_ols_slsqp.x)
        # print('para_differential_evolution', para_differential_evolution.x)
        alpha, beta = para_rmse_slsqp.x

    a = [sum(Y[:]) / float(len(Y))]
    b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
    y = [a[0] + b[0]]

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append(a[-1] + b[-1])

        a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        y.append(a[i + 1] + b[i + 1])

    rmse_index = rmse(Y[1:-fc], y[1:-fc-1])
    aic_index = aic(Y[1:-fc], y[1:-fc-1], 2)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc-1])])

    return {'predict': Y[-fc:], 'alpha': alpha, 'beta': beta, 'rmse': rmse_index, 'aic': aic_index, 'std_residual': std_residual, 'fittedvalues': y[1:-fc]}


def double_mul(x, fc, alpha=None, beta=None):
    """
        双参数指数平滑乘法预测模型
        parameters:
            --------
            x:时间序列数据列表
            fc:预测期数
            alpha:alpha参数,初始值默认为空
            beta:beta参数,初始默认值为空

        return:
            --------
            pred:预测值
            alpha:最优alpha值
            beta:最优beta值
            rmse
            aic
            std_residual:残差标准差
    """

    Y = x[:]

    if alpha is None or beta is None:

        initial_values = np.array([0.2]*2)
        boundaries = [(1/2*1e-1, 1-1/2*1e-1)]*2
        model = 'double_mul'

        # para_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model), method='SLSQP', jac='2-point',
        #                                bounds=boundaries)
        # alpha, beta = para_slsqp.x

        para_rmse_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model), method='SLSQP', jac='2-point',
                                            bounds=boundaries)
        # para_ols_slsqp = optimize.minimize(ols_loss_func, initial_values, args=(Y, model), method='SLSQP', jac='2-point',
        #                                    bounds=boundaries)
        # para_differential_evolution = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(
        #     Y, model), updating='deferred', workers=-1)
        print('rmse_loss_func', para_rmse_slsqp.x)
        # print('ols_loss_func', para_ols_slsqp.x)
        # print('para_differential_evolution', para_differential_evolution.x)
        alpha, beta = para_rmse_slsqp.x

    a = [sum(Y[:]) / float(len(Y))]
    b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
    y = [a[0] * b[0]]

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append(a[-1] * b[-1])

        a.append(alpha * Y[i] + (1 - alpha) * (a[i] * b[i]))
        b.append(beta * (a[i + 1] / a[i]) + (1 - beta) * b[i])
        y.append(a[i + 1] * b[i + 1])

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 2)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    return {'predict': Y[-fc:], 'alpha': alpha, 'beta': beta, 'rmse': rmse_index, 'aic': aic_index,
            'std_residual': std_residual, 'fittedvalues': y[1:-fc]}


def additive(x, m, fc, alpha=None, beta=None, gamma=None):
    """
        三参数指数平滑加法预测模型
        parameters:
            --------
            x:时间序列数据列表
            m:序列周期数(eg.序列为月份数据,m=12)
            fc:预测期数
            alpha:alpha参数,初始值默认为空
            beta:beta参数,初始值默认为空
            gamma:gamma参数,初始值默认为空

        return:
            --------
            pred:预测值
            alpha:最优alpha值
            beta:最优beta值
            gamma:最优gamma值
            rmse
            aic
            std_residual:残差标准差
    """

    Y = x[:]

    if alpha is None or beta is None or gamma is None:

        initial_values = np.array([0.2, 0.2, 2/3])
        boundaries = [(1/2*1e-1, 1-1/2*1e-1), (1/2*1e-1, 1-1/2*1e-1), (1e-1, 1-1e-1)]
        model = 'additive'

        para_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, m), method='SLSQP', jac='2-point',
                                       bounds=boundaries)
        alpha, beta, gamma = para_slsqp.x

        # para_rmse_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, m), method='SLSQP', jac='2-point',
        #                                     bounds=boundaries)
        # para_ols_slsqp = optimize.minimize(ols_loss_func, initial_values, args=(Y, model, m), method='SLSQP', jac='2-point',
        #                                    bounds=boundaries)
        # para_differential_evolution = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(
        #     Y, model, m), updating='deferred', workers=-1)
        # print('rmse_loss_func', para_rmse_slsqp.x)
        # print('ols_loss_func', para_ols_slsqp.x)
        # print('para_differential_evolution', para_differential_evolution.x)
        # alpha, beta, gamma = para_differential_evolution.x

    a = [sum(Y[:]) / float(len(Y))]
    b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
    s = (np.array(Y[:len(Y) // m * m]).reshape(-1, m).mean(axis=0) - a[0]).tolist()
    y = [a[0] + b[0] + np.mean(s)]

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append(round(a[-1] + b[-1] + s[-m], 4))

        a.append(alpha * (Y[i] - s[-m]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] - a[i + 1]) + (1 - gamma) * s[-m])
        y.append(a[i + 1] + b[i + 1] + s[1 - m])

    if not ((len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s) == m + len(x) + fc)):
        s = []
        print('分项长度有误，不能使用季节项序列')

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 2)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    return {'predict': Y[-fc:], 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'rmse': rmse_index, 'aic': aic_index,
            'std_residual': std_residual, 'max(s)': max(s), 'min(s)': min(s), 'mean(s)': np.mean(s), 'seasonal': s,
            'fittedvalues': y[1:-fc]}


def multiplicative(x, m, fc, alpha=None, beta=None, gamma=None):
    """
        三参数指数平滑乘法预测模型
        parameters:
            --------
            x:时间序列数据列表(日序列或周序列)
            m:序列周期数(eg.对于日序列年季节性m=365，日序列周季节性m=7，周序列年季节性m=52)
            fc:预测期数
            alpha:alpha参数,初始值默认为空
            beta:beta参数,初始值默认为空
            gamma:gamma参数,初始值默认为空

        return:
            --------
            pred:预测值
            alpha:最优alpha值
            beta:最优beta值
            gamma:最优gamma值
            rmse
            aic
            std_residual:残差标准差
    """
    Y = x[:]

    '''训练自变量alpha、beta、gamma'''
    if alpha is None or beta is None or gamma is None:

        # 给训练参数alpha、beta、gamma赋初值以启动梯度下降，并设置约束条件。
        initial_values = np.array([0.2]*3)
        boundaries = [(1/2*1e-1, 1-1/2*1e-1)]*3
        model = 'multiplicative'

        # 以最小二乘为目标函数训练参数，找到其全局最小值。
        para_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, m), method='SLSQP', jac='2-point',
                                       bounds=boundaries)
        alpha, beta, gamma = para_slsqp.x

        # para_rmse_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, m), method='SLSQP', jac='2-point',
        #                                     bounds=boundaries)
        # para_ols_slsqp = optimize.minimize(ols_loss_func, initial_values, args=(Y, model, m), method='SLSQP', jac='2-point',
        #                                    bounds=boundaries)
        # para_differential_evolution = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(
        #     Y, model, m), updating='deferred', workers=-1)
        # print('rmse_loss_func', para_rmse_slsqp.x)
        # print('ols_loss_func', para_ols_slsqp.x)
        # print('para_differential_evolution', para_differential_evolution.x)
        # alpha, beta, gamma = para_differential_evolution.x

    '''给截距a、斜率b、非线性项s、三参数乘法模型的目标函数y赋初值，以启动这四项的递归。
    a,b,s需要赋初值的原因是使基于递归表达式的模型在下面的循环中可以启动，并反映真实值的整体信息。
    应采用真实值在整个训练集上的均值(a[0])，斜率的均值(b[0])，平均偏离程度s[:m]，可最大程度避免随机波动和离群值的影响。'''
    a = [sum(Y[:]) / float(len(Y))]  # 整个训练集上真实值的均值
    '''训练集上前一半真实值和后一半真实值中每对点斜率的均值，len(Y)是奇偶数均可，此时len(Y)只需 >= m 即可'''
    b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
    '''真实值在整个训练集上所有完整周期的每个对应位置上的均值与整个训练集上真实值均值的比值，len(s)=m，即s的初始序列长度为所选周期长度m'''
    s = (np.array(Y[:len(Y) // m * m]).reshape(-1, m).mean(axis=0) / a[0]).tolist()
    '''人为给预测序列y赋初值，可使循环后a,b,y等长；此处也可令y=[]，则循环后预测序列y与真实值序列Y等长。即此处给y赋初值不是必要操作，y[0]不能使用。'''
    y = [(a[0] + b[0]) * np.mean(s)]

    '''模型外推，得到历史期的预测值y和预测期的预测值Y[-fc:]'''
    for i in range(len(Y) + fc):
        # s[]的索引中不能含有i-m，否则会出现第一个周期采用负索引，后面的周期采用正索引的情况；导致在正索引的第一个周期中，s本该取到序列中第二个周期的值，但却重复取到第一个周期的值。
        if i == len(Y):
            # 将Y.append()放在各分项的append之前，则输出的是离当期最近的fc个点的预测值；y最后一个点不能用，因为是距当期fc+1个点的预测值。
            # a、b序列的最后一个值不能用，即取到序列的倒数第二个值为止；而Y.append在a,b,s,y.append之前，所以a,b,s的索引为-1,-1,-m。
            Y.append((a[-1] + b[-1]) * s[-m])

        a.append(alpha * (Y[i] / s[-m]) + (1 - alpha) * (a[i] + b[i]))  # s序列取到的元素需要恰好比a,Y序列的元素在索引上向前推一个周期。
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] / a[i + 1]) + (1 - gamma) * s[-m])  # 右边s序列取到的元素需要恰好比a,Y,s.append后的序列的元素在索引上向前推一个周期。
        # a[i+1]，b[i+1]代表Y[i]的信息，用来计算y[i+1]；又因为y含有人为赋值但不可用初值y[0]，所以s[1-m]正好比y[i+1]向前推一个周期。
        y.append((a[i + 1] + b[i + 1]) * s[1 - m])

    if not ((len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s) == m + len(x) + fc)):
        s = []
        print('分项长度有误，不能使用季节项序列')

    '''预测精度指标，只需比较历史期，不用比较预测期上的指标，因为历史期上才有真实销量，预测期上无真实销量'''
    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 2)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    return {'predict': Y[-fc:], 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'rmse': rmse_index, 'aic': aic_index,
            'std_residual': std_residual, 'max(s)': max(s), 'min(s)': min(s), 'mean(s)': np.mean(s), 'seasonal': s,
            'fittedvalues': y[1:-fc]}


def multiseasonal_add_day(x, m_year=365, m_quarter=round(365/4), m_month=round(365/12), m_week=7, fc=4*7,
                          alpha=None, beta=None, gamma_year=None, gamma_quarter=None, gamma_month=None, gamma_week=None):
    """
        六参数指数平滑加法多重季节性日序列预测模型
        parameters:
            --------
            x:时间序列数据列表，日序列
            m:序列周期数(eg.年季节性时,m=365)
            fc:预测期数
            alpha:alpha参数,初始值默认为空
            beta:beta参数,初始值默认为空
            gamma:gamma参数,初始值默认为空

        return:
            --------
            pred:预测值
            alpha:最优alpha值
            beta:最优beta值
            gamma:最优gamma值
            rmse
            aic
            std_residual:残差标准差
    """

    Y = x[:]

    if alpha is None or beta is None or gamma_year is None or gamma_quarter is None or gamma_month is None or gamma_week is None:

        initial_values = np.array([.2]*(6-4) + [2/3]*4)
        boundaries = [(1e-2, 1-1e-2)]*(len(initial_values)-4) + [(1e-1, 1-1e-1)]*4
        model = 'multiseasonal_add_day'
        para_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month, m_week)),
                                       method='SLSQP', jac='2-point', bounds=boundaries)
        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = para_slsqp.x

        # para_rmse_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month, m_week)), method='SLSQP', jac='2-point',
        #                                     bounds=boundaries)
        # para_ols_slsqp = optimize.minimize(ols_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month, m_week)), method='SLSQP', jac='2-point',
        #                                    bounds=boundaries)
        # para_differential_evolution = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(
        # Y, model, (m_year, m_quarter, m_month, m_week)), updating='deferred', workers=-1)
        # alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = para_differential_evolution.x
        # print('rmse_loss_func', para_rmse_slsqp.x)
        # print('ols_loss_func', para_ols_slsqp.x)
        # print('para_differential_evolution', para_differential_evolution.x)

    a = [sum(Y[:]) / float(len(Y))]
    b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
    s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
    s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[0]).tolist()
    s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
    s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) - a[0]).tolist()
    y = [(a[0] + b[0]) + np.mean(s_year) + np.mean(s_quarter) + np.mean(s_month) + np.mean(s_week)]

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append((a[-1] + b[-1]) + s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month] + s_week[-m_week])

        a.append(alpha * (Y[i] - (s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month] + s_week[-m_week]))
                 + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s_year.append(gamma_year * (Y[i] - (a[i + 1] + s_quarter[-m_quarter] + s_month[-m_month] + s_week[-m_week]))
                      + (1 - gamma_year) * s_year[-m_year])
        s_quarter.append(gamma_quarter * (Y[i] - (a[i + 1] + s_year[-m_year] + s_month[-m_month] + s_week[-m_week]))
                      + (1 - gamma_quarter) * s_quarter[-m_quarter])
        s_month.append(gamma_month * (Y[i] - (a[i + 1] + s_year[-m_year] + s_quarter[-m_quarter] + s_week[-m_week]))
                      + (1 - gamma_month) * s_month[-m_month])
        s_week.append(gamma_week * (Y[i] - (a[i + 1] + s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month]))
                       + (1 - gamma_week) * s_week[-m_week])
        y.append((a[i + 1] + b[i + 1]) + s_year[1 - m_year] + s_quarter[1 - m_quarter] + s_month[1 - m_month] + s_week[1 - m_week])

    if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (len(s_quarter) == m_quarter + len(x) + fc) and (len(s_month) == m_month + len(x) + fc) and (len(s_week) == m_week + len(x) + fc):
        s_total = (np.array(s_year[m_year - m_week:]) + np.array(s_quarter[m_quarter - m_week:]) + np.array(
            s_month[m_month - m_week:]) + np.array(s_week[:])).tolist()
    else:
        s_total = []
        print('分项长度有误，不能使用季节项序列')

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 2)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    # return {'predict': Y[-fc:], 'alpha': alpha, 'beta': beta, 'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
    #         'gamma_month': gamma_month, 'gamma_week': gamma_week, 'rmse': rmse_index, 'aic': aic_index,
    #         'std_residual': std_residual, 'max(s)': max(s_total),
    #         'min(s)': min(s_total), 'seasonal': s_total}
    return {'len(s_year)': len(s_year), 'max(s_year)': max(s_year), 'min(s_year)': min(s_year),
            'mean(s_year)': np.mean(s_year), 'np.std(s_year)': np.std(s_year), 's_year75': np.percentile(s_year, 75),
            's_year25': np.percentile(s_year, 25),
            'len(s_quarter)': len(s_quarter), 'max(s_quarter)': max(s_quarter), 'min(s_quarter)': min(s_quarter),
            'mean(s_quarter)': np.mean(s_quarter), 'np.std(s_quarter)': np.std(s_quarter),
            's_quarter75': np.percentile(s_quarter, 75),
            's_quarter25': np.percentile(s_quarter, 25),
            'len(s_month)': len(s_month), 'max(s_month)': max(s_month), 'min(s_month)': min(s_month),
            'mean(s_month)': np.mean(s_month), 'np.std(s_month)': np.std(s_month),
            's_month75': np.percentile(s_month, 75),
            's_month25': np.percentile(s_month, 25),
            'len(s_week)': len(s_week), 'max(s_week)': max(s_week), 'min(s_week)': min(s_week),
            'mean(s_week)': np.mean(s_week), 'np.std(s_week)': np.std(s_week),
            's_week75': np.percentile(s_week, 75),
            's_week25': np.percentile(s_week, 25),
            'res': rmse_index - std_residual,
            'predict': Y[-fc:], 'alpha': alpha, 'beta': beta,
            'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
            'gamma_month': gamma_month, 'gamma_week': gamma_week,
            'rmse': rmse_index, 'aic': aic_index, 'std_residual': std_residual,
            'max(s_total)': max(s_total), 'min(s_total)': min(s_total), 'mean(s_total)': np.mean(s_total),
            's_total': s_total, 'fittedvalues': y[1:-fc]}


def multiseasonal_add_week(x, m_year=52, m_quarter=round(52/4), m_month=round(52/12), fc=4,
                          alpha=None, beta=None, gamma_year=None, gamma_quarter=None, gamma_month=None):
    """
        五参数指数平滑加法多重季节性周序列预测模型
        parameters:
            --------
            x:时间序列数据列表，周序列
            m:序列周期数(eg.年季节性时,m=52)
            fc:预测期数
            alpha:alpha参数,初始值默认为空
            beta:beta参数,初始值默认为空
            gamma:gamma参数,初始值默认为空

        return:
            --------
            pred:预测值
            alpha:最优alpha值
            beta:最优beta值
            gamma:最优gamma值
            rmse
            aic
            std_residual:残差标准差
    """

    Y = x[:]

    if alpha is None or beta is None or gamma_year is None or gamma_quarter is None or gamma_month is None:

        initial_values = np.array([.2]*(5-3) + [2/3]*3)
        boundaries = [(1e-2, 1-1e-2)]*(len(initial_values)-3) + [(1e-1, 1-1e-1)]*3
        model = 'multiseasonal_add_week'
        para_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month)),
                                       method='SLSQP', jac='2-point', bounds=boundaries)
        alpha, beta, gamma_year, gamma_quarter, gamma_month = para_slsqp.x

        # para_rmse_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month)), method='SLSQP', jac='2-point',
        #                                     bounds=boundaries)
        # para_ols_slsqp = optimize.minimize(ols_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month)), method='SLSQP', jac='2-point',
        #                                    bounds=boundaries)
        # para_differential_evolution = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(
        #     Y, model, (m_year, m_quarter, m_month)), updating='deferred', workers=-1)
        # alpha, beta, gamma_year, gamma_quarter, gamma_month = para_differential_evolution.x
        # print('rmse_loss_func', para_rmse_slsqp.x)
        # print('ols_loss_func', para_ols_slsqp.x)
        # print('para_differential_evolution', para_differential_evolution.x)

    a = [sum(Y[:]) / float(len(Y))]
    b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
    s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
    s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[0]).tolist()
    s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
    y = [(a[0] + b[0]) + np.mean(s_year) + np.mean(s_quarter) + np.mean(s_month)]

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append((a[-1] + b[-1]) + s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month])

        a.append(alpha * (Y[i] - (s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month]))
                 + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s_year.append(gamma_year * (Y[i] - (a[i + 1] + s_quarter[-m_quarter] + s_month[-m_month]))
                      + (1 - gamma_year) * s_year[-m_year])
        s_quarter.append(gamma_quarter * (Y[i] - (a[i + 1] + s_year[-m_year] + s_month[-m_month]))
                      + (1 - gamma_quarter) * s_quarter[-m_quarter])
        s_month.append(gamma_month * (Y[i] - (a[i + 1] + s_year[-m_year] + s_quarter[-m_quarter]))
                      + (1 - gamma_month) * s_month[-m_month])
        y.append((a[i + 1] + b[i + 1]) + s_year[1 - m_year] + s_quarter[1 - m_quarter] + s_month[1 - m_month])

    if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (len(s_quarter) == m_quarter + len(x) + fc) and (len(s_month) == m_month + len(x) + fc):
        s_total = (np.array(s_year[m_year - m_month:]) + np.array(s_quarter[m_quarter - m_month:]) + np.array(
            s_month[:])).tolist()
    else:
        s_total = []
        print('分项长度有误，不能使用季节项序列')

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 2)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    # return {'predict': Y[-fc:], 'alpha': alpha, 'beta': beta, 'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
    #         'gamma_month': gamma_month, 'rmse': rmse_index, 'aic': aic_index, 'std_residual': std_residual,
    #         'max(s)': max(s_total), 'min(s)': min(s_total),
    #         'seasonal': s_total}
    return {'len(s_year)': len(s_year), 'max(s_year)': max(s_year), 'min(s_year)': min(s_year),
            'mean(s_year)': np.mean(s_year), 'np.std(s_year)': np.std(s_year), 's_year75': np.percentile(s_year, 75),
            's_year25': np.percentile(s_year, 25),
            'len(s_quarter)': len(s_quarter), 'max(s_quarter)': max(s_quarter), 'min(s_quarter)': min(s_quarter),
            'mean(s_quarter)': np.mean(s_quarter), 'np.std(s_quarter)': np.std(s_quarter),
            's_quarter75': np.percentile(s_quarter, 75),
            's_quarter25': np.percentile(s_quarter, 25),
            'len(s_month)': len(s_month), 'max(s_month)': max(s_month), 'min(s_month)': min(s_month),
            'mean(s_month)': np.mean(s_month), 'np.std(s_month)': np.std(s_month),
            's_month75': np.percentile(s_month, 75),
            's_month25': np.percentile(s_month, 25),
            'res': rmse_index - std_residual,
            'predict': Y[-fc:], 'alpha': alpha, 'beta': beta,
            'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
            'gamma_month': gamma_month, 'rmse': rmse_index, 'aic': aic_index, 'std_residual': std_residual,
            'max(s_total)': max(s_total), 'min(s_total)': min(s_total), 'mean(s_total)': np.mean(s_total),
            's_total': s_total, 'fittedvalues': y[1:-fc]}


def multiseasonal_mul_day(x, m_year=365, m_quarter=round(365/4), m_month=round(365/12), m_week=7, fc=4*7,
                          alpha=None, beta=None, gamma_year=None, gamma_quarter=None, gamma_month=None, gamma_week=None):
    """
        六参数指数平滑乘法多重季节性日序列预测模型
        parameters:
            --------
            x:时间序列数据列表，日序列
            m:序列周期数(eg.年季节性时,m=365)
            fc:预测期数
            alpha:alpha参数,初始值默认为空
            beta:beta参数,初始值默认为空
            gamma:gamma参数,初始值默认为空

        return:
            --------
            pred:预测值
            alpha:最优alpha值
            beta:最优beta值
            gamma:最优gamma值
            rmse
            aic
            std_residual:残差标准差
    """

    Y = x[:]

    if alpha is None or beta is None or gamma_year is None or gamma_quarter is None or gamma_month is None or gamma_week is None:

        initial_values = np.array([.2]*6)
        boundaries = [(1/2*1e-1, 1-1/2*1e-1)]*6
        model = 'multiseasonal_mul_day'
        para_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month, m_week)), method='SLSQP', jac='2-point',
                                       bounds=boundaries)
        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = para_slsqp.x
        # para_rmse_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month, m_week)), method='SLSQP', jac='2-point',
        #                                     bounds=boundaries)
        # para_differential_evolution_rmse = optimize.differential_evolution(rmse_loss_func, bounds=boundaries, args=(
        #                                                                     Y, model, (m_year, m_quarter, m_month, m_week)), updating='deferred', workers=-1)
        # para_ols_slsqp = optimize.minimize(ols_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month, m_week)), method='SLSQP', jac='2-point',
        #                                    bounds=boundaries)
        # para_differential_evolution = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(Y, model, (m_year, m_quarter, m_month, m_week)), updating='deferred', workers=-1)
        # para_shgo = optimize.shgo(ols_loss_func, bounds=boundaries, args=(Y, model, (m_year, m_quarter, m_month, m_week)), iters=3)
        # alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = para_differential_evolution.x
        # print('rmse_loss_func', para_rmse_slsqp.x)
        # print('ols_loss_func', para_ols_slsqp.x)
        # print('differential_evolution-rmse', para_differential_evolution_rmse.x)
        # print('differential_evolution-ols', para_differential_evolution.x)
        # print('shgo-ols', para_shgo.x)

    a = [sum(Y[:]) / float(len(Y))]
    b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
    s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
    s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[0]).tolist()
    s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
    s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) / a[0]).tolist()
    y = [(a[0] + b[0]) * np.mean(s_year) * np.mean(s_quarter) * np.mean(s_month) * np.mean(s_week)]

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append((a[-1] + b[-1]) * s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month] * s_week[-m_week])

        a.append(alpha * (Y[i] / (s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month] * s_week[-m_week]))
                 + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s_year.append(gamma_year * (Y[i] / (a[i + 1] * s_quarter[-m_quarter] * s_month[-m_month] * s_week[-m_week]))
                      + (1 - gamma_year) * s_year[-m_year])
        s_quarter.append(gamma_quarter * (Y[i] / (a[i + 1] * s_year[-m_year] * s_month[-m_month] * s_week[-m_week]))
                      + (1 - gamma_quarter) * s_quarter[-m_quarter])
        s_month.append(gamma_month * (Y[i] / (a[i + 1] * s_year[-m_year] * s_quarter[-m_quarter] * s_week[-m_week]))
                      + (1 - gamma_month) * s_month[-m_month])
        s_week.append(gamma_week * (Y[i] / (a[i + 1] * s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month]))
                       + (1 - gamma_week) * s_week[-m_week])
        y.append((a[i + 1] + b[i + 1]) * s_year[1 - m_year] * s_quarter[1 - m_quarter] * s_month[1 - m_month] * s_week[1 - m_week])

    if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (len(s_quarter) == m_quarter + len(x) + fc) and (len(s_month) == m_month + len(x) + fc) and (len(s_week) == m_week + len(x) + fc):
        s_total = (np.array(s_year[m_year - m_week:]) * np.array(s_quarter[m_quarter - m_week:]) * np.array(
            s_month[m_month - m_week:]) * np.array(s_week[:])).tolist()
    else:
        s_total = []
        print('分项长度有误，不能使用季节项序列')

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 2)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    # return {'predict': Y[-fc:], 'alpha': alpha, 'beta': beta, 'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
    #         'gamma_month': gamma_month, 'gamma_week': gamma_week, 'rmse': rmse_index, 'aic': aic_index,
    #         'std_residual': std_residual, 'max(s)': max(s_total),
    #         'min(s)': min(s_total),
    #         'seasonal': s_total}
    return {'len(s_year)': len(s_year), 'max(s_year)': max(s_year), 'min(s_year)': min(s_year),
            'mean(s_year)': np.mean(s_year), 'np.std(s_year)': np.std(s_year), 's_year75': np.percentile(s_year, 75),
            's_year25': np.percentile(s_year, 25),
            'len(s_quarter)': len(s_quarter), 'max(s_quarter)': max(s_quarter), 'min(s_quarter)': min(s_quarter),
            'mean(s_quarter)': np.mean(s_quarter), 'np.std(s_quarter)': np.std(s_quarter),
            's_quarter75': np.percentile(s_quarter, 75),
            's_quarter25': np.percentile(s_quarter, 25),
            'len(s_month)': len(s_month), 'max(s_month)': max(s_month), 'min(s_month)': min(s_month),
            'mean(s_month)': np.mean(s_month), 'np.std(s_month)': np.std(s_month),
            's_month75': np.percentile(s_month, 75),
            's_month25': np.percentile(s_month, 25),
            'len(s_week)': len(s_week), 'max(s_week)': max(s_week), 'min(s_week)': min(s_week),
            'mean(s_week)': np.mean(s_week), 'np.std(s_week)': np.std(s_week),
            's_week75': np.percentile(s_week, 75),
            's_week25': np.percentile(s_week, 25),
            'res': rmse_index - std_residual,
            'predict': Y[-fc:], 'alpha': alpha, 'beta': beta,
            'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
            'gamma_month': gamma_month, 'gamma_week': gamma_week,
            'rmse': rmse_index, 'aic': aic_index, 'std_residual': std_residual,
            'max(s_total)': max(s_total), 'min(s_total)': min(s_total), 'mean(s_total)': np.mean(s_total),
            's_total': s_total, 'fittedvalues': y[1:-fc]}


def multiseasonal_mul_week(x, m_year=52, m_quarter=round(52/4), m_month=round(52/12), fc=4,
                          alpha=None, beta=None, gamma_year=None, gamma_quarter=None, gamma_month=None):
    """
        五参数指数平滑乘法多重季节性周序列预测模型
        parameters:
            --------
            x:时间序列数据列表，周序列
            m:序列周期数(eg.年季节性,m=52)
            fc:预测期数
    """

    Y = x[:]

    if alpha is None or beta is None or gamma_year is None or gamma_quarter is None or gamma_month is None:

        initial_values = np.array([.2]*(5-3) + [2/3]*3)
        boundaries = [(1e-2, 1-1e-2)]*(len(initial_values)-3) + [(1e-1, 1-1e-1)]*3
        model = 'multiseasonal_mul_week'

        para_rmse_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month)),
                                            method='SLSQP', jac='2-point',
                                            bounds=boundaries)
        alpha, beta, gamma_year, gamma_quarter, gamma_month = para_rmse_slsqp.x

        # para_ols_slsqp = optimize.minimize(ols_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month)),
        #                                    method='SLSQP', jac='2-point',
        #                                    bounds=boundaries)
        # para_differential_evolution = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(
        #     Y, model, (m_year, m_quarter, m_month)), updating='deferred', workers=-1)
        # alpha, beta, gamma_year, gamma_quarter, gamma_month = para_differential_evolution.x
        # print('rmse_loss_func', para_rmse_slsqp.x)
        # print('ols_loss_func', para_ols_slsqp.x)
        # print('para_differential_evolution', para_differential_evolution.x)

    a = [sum(Y[:]) / float(len(Y))]
    b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
    s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
    s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[0]).tolist()
    s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
    y = [(a[0] + b[0]) * np.mean(s_year) * np.mean(s_quarter) * np.mean(s_month)]

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append((a[-1] + b[-1]) * s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month])

        a.append(alpha * (Y[i] / (s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month]))
                 + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s_year.append(gamma_year * (Y[i] / (a[i + 1] * s_quarter[-m_quarter] * s_month[-m_month]))
                      + (1 - gamma_year) * s_year[-m_year])
        s_quarter.append(gamma_quarter * (Y[i] / (a[i + 1] * s_year[-m_year] * s_month[-m_month]))
                      + (1 - gamma_quarter) * s_quarter[-m_quarter])
        s_month.append(gamma_month * (Y[i] / (a[i + 1] * s_year[-m_year] * s_quarter[-m_quarter]))
                      + (1 - gamma_month) * s_month[-m_month])
        y.append((a[i + 1] + b[i + 1]) * s_year[1 - m_year] * s_quarter[1 - m_quarter] * s_month[1 - m_month])

    if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (len(s_quarter) == m_quarter + len(x) + fc) and (len(s_month) == m_month + len(x) + fc):
        s_total = (np.array(s_year[m_year - m_month:]) * np.array(s_quarter[m_quarter - m_month:]) * np.array(
            s_month[:])).tolist()
    else:
        s_total = []
        print('分项长度有误，不能使用季节项序列')

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 2)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    # return {'predict': Y[-fc:], 'alpha': alpha, 'beta': beta, 'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
    #         'gamma_month': gamma_month, 'rmse': rmse_index, 'aic': aic_index, 'std_residual': std_residual,
    #         'max(s)': max(s_total),
    #         'min(s)': min(s_total),
    #         'seasonal': s_total}
    return {'len(s_year)': len(s_year), 'max(s_year)': max(s_year), 'min(s_year)': min(s_year),
            'mean(s_year)': np.mean(s_year), 'np.std(s_year)': np.std(s_year), 's_year75': np.percentile(s_year, 75),
            's_year25': np.percentile(s_year, 25),
            'len(s_quarter)': len(s_quarter), 'max(s_quarter)': max(s_quarter), 'min(s_quarter)': min(s_quarter),
            'mean(s_quarter)': np.mean(s_quarter), 'np.std(s_quarter)': np.std(s_quarter),
            's_quarter75': np.percentile(s_quarter, 75),
            's_quarter25': np.percentile(s_quarter, 25),
            'len(s_month)': len(s_month), 'max(s_month)': max(s_month), 'min(s_month)': min(s_month),
            'mean(s_month)': np.mean(s_month), 'np.std(s_month)': np.std(s_month),
            's_month75': np.percentile(s_month, 75),
            's_month25': np.percentile(s_month, 25),
            'res': rmse_index - std_residual,
            'predict': Y[-fc:], 'alpha': alpha, 'beta': beta,
            'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
            'gamma_month': gamma_month, 'rmse': rmse_index, 'aic': aic_index, 'std_residual': std_residual,
            'max(s_total)': max(s_total), 'min(s_total)': min(s_total), 'mean(s_total)': np.mean(s_total),
            's_total': s_total, 'fittedvalues': y[1:-fc]}

# comoare_func需调整，因为AIC是经验公式，其不同形式适用于不同条件下两条序列近似度的比较，可靠性低于属于理论公式的RMSE，不应作为首要判断条件。
# 而RMSE会将离群点的效果放大，我们恰好不希望预测值与真实值偏差很大的情况出现，所以更适合。可重写判断函数，将各种指标都考虑进来。
# 但最好是将所有参与训练的平滑类模型的预测序列输出，进入后续步骤，不进行优选。


# n = 100
# np.random.seed(500)
if __name__ == "__main__":
    day = list(np.random.randint(1, 10, 730))
    week = list(np.random.randint(1, 10, 104))
    a = simple(day, 4)
    print('simple', a, '\n')
    b = linear(day, 4)
    print('linear', b, '\n')
    c = double_mul(day, 4)
    print('double_mul', c, '\n')
    d = additive(day, 365, 4)
    print('additive', d, '\n')
    e = multiplicative(day, 365, 4)
    print('multiplicative', e, '\n')
    g1 = multiseasonal_add_day(day)
    print('multiseasonal_add_day', g1, '\n')
    g2 = multiseasonal_add_week(week)
    print('multiseasonal_add_week', g2, '\n')
    g3 = multiseasonal_mul_day(day)
    print('multiseasonal_mul_day', g3, '\n')
    g4 = multiseasonal_mul_week(week)
    print('multiseasonal_mul_week', g4)
