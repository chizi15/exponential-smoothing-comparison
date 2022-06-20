# coding:utf-8

from __future__ import division
import numpy as np
from sys import exit
from scipy import optimize
import math

"""
特别重要：
当序列长度为周期的n倍或n倍多几个点时，例如周期365，序列长度730，则从第730个点，即最后一个拟合点开始，包括其后数个预测点，会出现突变，
越远离第730个点，突变量越小，越接近越大；当序列长度比周期的倍数n越大时，这种突变影响越小；对于本例，即靠近730的最近的几个点突变量较大，
所以序列长度应比730多一些点如760。综上，序列长度应为周期的n倍多一些点。
"""
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


def aic(actuallist, fittedvalues, k):
    """
        aic判定准则
        parameters:
            --------
            k:模型参数数量(eg.linear函数有alpha,beta参数,k为2)
    """
    actual_array = np.array(actuallist)
    fittedvalues_array = np.array(fittedvalues)

    n = len(actual_array)  # 序列长度
    rss = sum((fittedvalues_array - actual_array) ** 2)  # 残差平方和
    aic = n * np.log(rss / n) + 2 * k

    return aic


def rmse_loss_func(params, *args):
    """
        构造平滑类模型的rmse目标函数，params是自变量，return返回的是因变量，args是需要外部输入的超参数，优化算法作用于该目标函数。

        计算均方根误差
        parameters:
            --------
            params:该元组包含自变量 alpha, beta, gamma, gamma_year, gamma_quarter, gamma_month, gamma_week 的初始值。
            args:该元组包含原始计算序列 Y(args[0])、模型 type(args[1])、周期 m, m_year, m_quarter, m_month, m_week(args[2])。

        return:
            --------
            rmse:原始序列与预测序列间的均方根误差，作为目标函数。
    """

    Y = args[0]
    model = args[1]

    if model == 'simple':
        """
            单参数模型，适用于预测无趋势无季节性序列，其预测结果为一水平直线。
            alpha是自变量，rmse(Y[1:], y[1:-1])是目标函数。
            在Rmse中alpha的最高次数是一次，所以此时Rmse为凸函数，局部优化和全局优化算法的解相同。
        """
        alpha = params  # 将alpha定义为自变量
        '''给截距序列a及预测序列y赋初值'''
        a = [sum(Y[:]) / float(len(Y))]
        y = [a[0]]

        # 构造截距序列a及预测序列y
        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * a[i])  # 此处的a与y均是预测序列
            y.append(a[i+1])

    elif model == 'linear':
        '''
            alpha、beta是自变量，rmse(Y[1:], y[1:-1])是目标函数；在Rmse中alpha、beta的最高次数是一次，所以此时Rmse为凸函数，局部优化和全局优化算法的解相同。
        '''
        alpha, beta = params
        # 给截距序列a、斜率序列b、预测序列y赋初值
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        y = [a[0] + b[0]]

        # 构造截距level序列a、斜率trend序列b、预测序列y
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
        y = [(a[0] + b[0]) + s[0]]

        for i in range(len(Y)):
            a.append(alpha * (Y[i] - s[-m]) + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s.append(gamma * (Y[i] - a[i + 1]) + (1 - gamma) * s[-m])
            y.append(a[i + 1] + b[i + 1] + s[-m])

    elif model == 'multiplicative':

        alpha, beta, gamma = params
        m = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s = (np.array(Y[:len(Y) // m * m]).reshape(-1, m).mean(axis=0) / a[0]).tolist()
        y = [(a[0] + b[0]) * s[0]]

        for i in range(len(Y)):
            a.append(alpha * (Y[i] / s[-m]) + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s.append(gamma * (Y[i] / a[i + 1]) + (1 - gamma) * s[-m])
            y.append((a[i + 1] + b[i + 1]) * s[-m])

    elif model == 'multi_seasonal_add_day':

        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = params
        m_year, m_quarter, m_month, m_week = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
        s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) - a[0]).tolist()
        y = [a[0] + b[0] + s_year[0] + s_quarter[0] + s_month[0] + s_week[0]]

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
            y.append(a[i + 1] + b[i + 1] + s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month] + s_week[-m_week])

    elif model == 'multi_seasonal_add_week':

        alpha, beta, gamma_year, gamma_quarter, gamma_month = params
        m_year, m_quarter, m_month = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
        y = [(a[0] + b[0]) + s_year[0] + s_quarter[0] + s_month[0]]

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
            y.append(a[i + 1] + b[i + 1] + s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month])

    elif model == 'multi_seasonal_mul_day':

        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = params
        m_year, m_quarter, m_month, m_week = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
        s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) / a[0]).tolist()
        y = [(a[0] + b[0]) * s_year[0] * s_quarter[0] * s_month[0] * s_week[0]]

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
            y.append((a[i + 1] + b[i + 1]) * s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month] *
                     s_week[-m_week])

    elif model == 'multi_seasonal_mul_week':

        alpha, beta, gamma_year, gamma_quarter, gamma_month = params
        m_year, m_quarter, m_month = args[2]
        a = [sum(Y) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
        y = [(a[0] + b[0]) * s_year[0] * s_quarter[0] * s_month[0]]  # y[0]应代表真实序列Y的平均信息，但不会用到。

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
            y.append((a[i + 1] + b[i + 1]) * s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month])

    else:
        exit('args[1] must be one of smoothing models')
    '''
        各分项的初始值采用全体真实值的平均信息后，预测值y[0]也代表平均信息，与Y[0]不再具备对应关系；而预测值y[1]是由真实值Y[0]等向后推出，
        则y[1]应与Y[1]匹配；即y[n+1]是由Y[n]算出，则y[n+1]与Y[n+1]匹配。又因为y具有人为赋初值y[0]，len(y)==len(Y+1)，Y[-1]产生预测值y[-1],
        则没有真实值与y[-1]匹配，则最后一个真实值Y[-1]与倒数第二个预测值y[-2]匹配。综上，匹配关系为：Y[1:], y[1:-1]。
    '''
    rmse_distance = rmse(Y[1:], y[1:-1])  # 构造RMSE目标函数

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
        alpha = params  # 将alpha定义为自变量
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
        # y = [(a[0] + b[0]) + np.mean(s)]
        y = [(a[0] + b[0]) + s[0]]

        for i in range(len(Y)):
            a.append(alpha * (Y[i] - s[-m]) + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s.append(gamma * (Y[i] - a[i + 1]) + (1 - gamma) * s[-m])
            y.append(a[i + 1] + b[i + 1] + s[-m])

    elif model == 'multiplicative':

        alpha, beta, gamma = params
        m = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s = (np.array(Y[:len(Y) // m * m]).reshape(-1, m).mean(axis=0) / a[0]).tolist()
        # y = [(a[0] + b[0]) * np.mean(s)]
        y = [(a[0] + b[0]) * s[0]]

        for i in range(len(Y)):
            a.append(alpha * (Y[i] / s[-m]) + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s.append(gamma * (Y[i] / a[i + 1]) + (1 - gamma) * s[-m])
            y.append((a[i + 1] + b[i + 1]) * s[-m])

    elif model == 'multi_seasonal_add_day':

        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = params
        m_year, m_quarter, m_month, m_week = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
        s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) - a[0]).tolist()
        y = [(a[0] + b[0]) + s_year[0] + s_quarter[0] + s_month[0] + s_week[0]]

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
            y.append((a[i + 1] + b[i + 1]) + s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month] +
                     s_week[-m_week])

    elif model == 'multi_seasonal_add_week':

        alpha, beta, gamma_year, gamma_quarter, gamma_month = params
        m_year, m_quarter, m_month = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
        y = [(a[0] + b[0]) + s_year[0] + s_quarter[0] + s_month[0]]

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
            y.append((a[i + 1] + b[i + 1]) + s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month])

    elif model == 'multi_seasonal_mul_day':

        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = params
        m_year, m_quarter, m_month, m_week = args[2]
        a = [sum(Y[:]) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
        s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) / a[0]).tolist()
        y = [(a[0] + b[0]) * s_year[0] * s_quarter[0] * s_month[0] * s_week[0]]

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
            y.append((a[i + 1] + b[i + 1]) * s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month] *
                     s_week[-m_week])

    elif model == 'multi_seasonal_mul_week':

        alpha, beta, gamma_year, gamma_quarter, gamma_month = params
        m_year, m_quarter, m_month = args[2]
        a = [sum(Y) / float(len(Y))]
        b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
        y = [(a[0] + b[0]) * s_year[0] * s_quarter[0] * s_month[0]]  # y[0]应代表真实序列Y的平均信息，但不会用到。

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
            y.append((a[i + 1] + b[i + 1]) * s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month])

    else:
        exit('Type must be in one of smoothing models')

    '''
        各分项的初始值采用全体真实值的平均信息后，预测值y[0]也代表平均信息，与Y[0]不再具备对应关系；而预测值y[1]是由真实值Y[0]等向后推出，
        则y[1]应与Y[1]匹配；即y[n+1]是由Y[n]算出，则y[n+1]与Y[n+1]匹配。又因为y具有人为赋初值y[0]，len(y)==len(Y+1)，Y[-1]产生预测值y[-1],
        则没有真实值与y[-1]匹配，则最后一个真实值Y[-1]与倒数第二个预测值y[-2]匹配。综上，匹配关系为：Y[1:], y[1:-1]。
    '''
    ols_distance = ols(Y[1:], y[1:-1])  # 因为y[-1]是第一个预测值，已不属于拟合区间，所以不在目标函数取值范围中。

    return ols_distance


def loss_func(params, *args):
    """
        构造平滑类模型的rmse目标函数，params是自变量，return返回的是因变量，args是需要外部输入的超参数，优化算法作用于该目标函数。

        计算均方根误差
        parameters:
            --------
            params:该元组包含自变量 alpha, beta, gamma, gamma_year, gamma_quarter, gamma_month, gamma_week 的初始值。
            args:该元组包含原始计算序列 Y(args[0])、模型 type(args[1])、周期 m, m_year, m_quarter, m_month, m_week(args[2])。

        return:
            --------
            rmse:原始序列与预测序列间的均方根误差，作为目标函数。
    """

    # alpha, beta, gamma_year, gamma_week, gamma_month, gamma_quarter = params
    Y = args[0]
    model = args[1]
    a = [sum(Y) / float(len(Y))]
    b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
    y = [1]

    if model == 'simple':
        """
            单参数模型，适用于预测无趋势无季节性序列，其预测结果为一水平直线。
            alpha是自变量，rmse(Y[1:], y[1:-1])是目标函数。
            在Rmse中alpha的最高次数是一次，所以此时Rmse为凸函数，局部优化和全局优化算法的解相同。
        """
        alpha = params  # 将alpha定义为自变量
        # '''给截距序列a及预测序列y赋初值'''
        # a = [sum(Y[:]) / float(len(Y))]
        # y = [a[0]]

        # 构造截距序列a及预测序列y
        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * a[i])  # 此处的a与y均是预测序列
            y.append(a[i+1])

    elif model == 'linear':
        '''
            alpha、beta是自变量，rmse(Y[1:], y[1:-1])是目标函数；在Rmse中alpha、beta的最高次数是一次，所以此时Rmse为凸函数，局部优化和全局优化算法的解相同。
        '''
        alpha, beta = params
        # # 给截距序列a、斜率序列b、预测序列y赋初值
        # a = [sum(Y[:]) / float(len(Y))]
        # b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        # y = [a[0] + b[0]]

        # 构造截距level序列a、斜率trend序列b、预测序列y
        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            y.append(a[i + 1] + b[i + 1])

    elif model == 'double_mul':
        '''
            alpha、beta是自变量，rmse(Y[1:], y[1:-1])是目标函数；在Rmse中alpha、beta的最高次数是一次，所以此时Rmse为凸函数，局部优化和全局优化算法的解相同。
        '''
        alpha, beta = params
        # # 给截距序列a、斜率序列b、预测序列y赋初值
        # a = [sum(Y[:]) / float(len(Y))]
        # b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        # y = [a[0] * b[0]]

        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * (a[i] * b[i]))
            b.append(beta * (a[i + 1] / a[i]) + (1 - beta) * b[i])
            y.append(a[i + 1] * b[i + 1])

    elif model == 'additive':

        alpha, beta, gamma = params
        m_year, m_week, m_month, m_quarter = args[3]
        # a = [sum(Y[:]) / float(len(Y))]
        # b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s = (np.array(Y[:len(Y) // m * m]).reshape(-1, m).mean(axis=0) - a[0]).tolist()
        # y = [(a[0] + b[0]) + s[0]]

        for i in range(len(Y)):
            a.append(alpha * (Y[i] - s[-m]) + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s.append(gamma * (Y[i] - a[i + 1]) + (1 - gamma) * s[-m])
            y.append(a[i + 1] + b[i + 1] + s[-m])

    elif model == 'multiplicative':

        alpha, beta, gamma = params
        m_year, m_week, m_month, m_quarter = args[3]
        # a = [sum(Y[:]) / float(len(Y))]
        # b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s = (np.array(Y[:len(Y) // m * m]).reshape(-1, m).mean(axis=0) / a[0]).tolist()
        # y = [(a[0] + b[0]) * s[0]]

        for i in range(len(Y)):
            a.append(alpha * (Y[i] / s[-m]) + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s.append(gamma * (Y[i] / a[i + 1]) + (1 - gamma) * s[-m])
            y.append((a[i + 1] + b[i + 1]) * s[-m])

    elif model == 'multi_seasonal_add_day':

        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = params
        m_year, m_week, m_month, m_quarter = args[3]
        # a = [sum(Y[:]) / float(len(Y))]
        # b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
        s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) - a[0]).tolist()
        # y = [(a[0] + b[0]) + s_year[0] + s_quarter[0] + s_month[0] + s_week[0]]

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
            y.append(a[i + 1] + b[i + 1] + s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month] + s_week[-m_week])

    elif model == 'multi_seasonal_add_week':

        alpha, beta, gamma_year, gamma_quarter, gamma_month = params
        m_year, m_week, m_month, m_quarter = args[3]
        # a = [sum(Y[:]) / float(len(Y))]
        # b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
        # y = [(a[0] + b[0]) + s_year[0] + s_quarter[0] + s_month[0]]

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
            y.append(a[i + 1] + b[i + 1] + s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month])

    elif model == 'multi_seasonal_mul_day':

        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = params
        m_year, m_week, m_month, m_quarter = args[3]
        # a = [sum(Y[:]) / float(len(Y))]
        # b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
        s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) / a[0]).tolist()
        # y = [(a[0] + b[0]) * s_year[0] * s_quarter[0] * s_month[0] * s_week[0]]

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
            y.append((a[i + 1] + b[i + 1]) * s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month] *
                     s_week[-m_week])

    elif model == 'multi_seasonal_mul_week':

        alpha, beta, gamma_year, gamma_quarter, gamma_month = params
        m_year, m_week, m_month, m_quarter = args[3]
        # a = [sum(Y) / float(len(Y))]
        # b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
        s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
        s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[
            0]).tolist()
        s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
        # y = [(a[0] + b[0]) * s_year[0] * s_quarter[0] * s_month[0]]  # y[0]应代表真实序列Y的平均信息，但不会用到。

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
            y.append((a[i + 1] + b[i + 1]) * s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month])

    else:
        exit('args[1] must be one of smoothing models')
    '''
        各分项的初始值采用全体真实值的平均信息后，预测值y[0]也代表平均信息，与Y[0]不再具备对应关系；而预测值y[1]是由真实值Y[0]等向后推出，
        则y[1]应与Y[1]匹配；即y[n+1]是由Y[n]算出，则y[n+1]与Y[n+1]匹配。又因为y具有人为赋初值y[0]，len(y)==len(Y+1)，Y[-1]产生预测值y[-1],
        则没有真实值与y[-1]匹配，则最后一个真实值Y[-1]与倒数第二个预测值y[-2]匹配。综上，匹配关系为：Y[1:], y[1:-1]。
    '''
    if args[2] == 'RMSE':
        distance = rmse(Y[1:], y[1:-1])  # 构造rmse目标函数
    elif args[2] == 'OLS':
        distance = ols(Y[1:], y[1:-1])  # 构造ols目标函数
    else:
        raise Exception('type of loss function must be either \'RMSE\' or \'OLS\'')
    return distance


def simple(x, fc, alpha=None, boundaries = [(1 / 2 * 1e-1, 1 - 1 / 2 * 1e-1)], initial_values = np.array([0.2]),
           method='L-BFGS-B', weights=None):
    """
        单参数指数平滑模型
        parameters:
            --------
            x: 时间序列数据列表
            fc: 预测期数
            alpha: alpha 参数,初始值默认为空
            boundaries：参数 alpha 训练时的边界值
            initial_values：参数 alpha 训练时使用的初始值，最好使用同序列上一次训练所得参数值，大概率可减小梯度下降的迭代次数
            method：目标函数做梯度下降时的方法，优化算法可选['L-BFGS-B','SLSQP','TNC','trust-constr','global']
            weights: 对输入模型的历史值设置权重，用于给各迭代项的首项赋值；可为None，则越近的点权重越高，越远的点权重越低；可为equal，则等权重；
            或其他任何字符，则无权重，即a[0]=Y[0]

        return:
            --------
            predict:一个预测期的预测值
            alpha: level 项的非训练指定参数，或经训练的最优参数
            rmse：计算历史期真实值和拟合值的均方根误差，与小越好
            aic：计算历史期真实值和拟合值的信息准则，判定拟合的优良程度，越小越好
            std_residual:真实序列与预测序列残差的标准差，越小越好
            fittedvalues：历史区间的拟合值
    """

    Y = x[:]
    if alpha is None:
        model = 'simple'
        if method != 'global':
            paras = optimize.minimize(loss_func, initial_values, args=(Y, model, 'RMSE'), method=method, jac='2-point',
                                            bounds=boundaries)
        else:
            paras = optimize.differential_evolution(loss_func, bounds=boundaries, args=(Y, model, 'OLS'),
                                                    updating='deferred', workers=-1)
        alpha = float(paras.x)
    else:
        if alpha < boundaries[0][0] or alpha > boundaries[0][1]:
            raise Exception('参数初始值超限')
    print('paras', alpha)

    if weights is None:
        weights = []
        for i in range(1, len(Y) + 1):
            weights.append(i)
        weights = np.array(weights) / sum(weights)
        a = [np.average(Y, weights=weights)]
        y = [a[0]]
    elif weights == 'equal':
        weights = np.array([1 / len(Y)] * len(Y))
        a = [np.average(Y, weights=weights)]
        y = [a[0]]
    else:
        weights = np.array([1 / len(Y)] * len(Y))
        a = [Y[0]]
        y = [a[0]]

    for i in range(len(Y) + fc):
        if i == len(Y):
            Y.append(a[-1])
        a.append(alpha * Y[i] + (1 - alpha) * a[i])
        y.append(a[i+1])

    if sum(np.array(Y[-fc:]) - np.array(y[-fc-1:-1])) != 0:
        print(Y[-fc:], '\n', y[-fc-1:-1])
        raise Exception('预测值Y和y不相等')

    rmse_index = rmse(Y[1:-fc], y[1:-fc-1])
    aic_index = aic(Y[1:-fc], y[1:-fc-1], 1)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc-1])])  # 历史区间内残差序列的标准差

    return {'predict': y[-fc-1: -1], 'alpha': alpha, 'rmse': rmse_index, 'aic': aic_index, 'std_residual': std_residual,
            'fittedvalues': y[:-fc-1], 'weights': weights}


def linear(x, fc, alpha=None, beta=None, boundaries = [(1 / 2 * 1e-1, 1 - 1 / 2 * 1e-1)] * 2, method='SLSQP',
           initial_values = np.array([0.2] * 2), weights=None):
    """
        双参数指数平滑乘法模型
        parameters:
            --------
            x:时间序列数据列表
            fc:预测期数
            alpha: level 项 a 的参数,初始值默认为空
            beta: trend 项 b 的参数,初始值默认为空
            boundaries：参数 alpha，beta 训练时的边界值
            initial_values：参数 alpha，beta 训练时使用的初始值，最好使用同序列上一次训练所得参数值，大概率可减小梯度下降的迭代次数
            method：目标函数做梯度下降时的方法，优化算法可选['L-BFGS-B','SLSQP','TNC','trust-constr','global']
            weights:对输入模型的历史值设置权重，用于给各迭代项的首项赋值；可为None，则越近的点权重越高，越远的点权重越低；可为equal，则等权重；
            或其他任何字符，则无权重，即a[0]=Y[0],b[0]=Y[1]-Y[0]

        return:
            --------
            predict:一个预测期的预测值
            alpha: level 项的非训练指定参数，或经训练的最优参数
            beta: trend 项的非训练指定参数，或经训练的最优参数
            rmse：计算历史期真实值和拟合值的均方根误差，与小越好
            aic：计算历史期真实值和拟合值的信息准则，判定拟合的优良程度，越小越好
            std_residual:真实序列与预测序列残差的标准差，越小越好
            fittedvalues：历史区间的拟合值
    """

    Y = x[:]
    if alpha is None or beta is None:
        model = 'linear'
        if method != 'global':
            para = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model), method=method, jac='2-point',
                                            bounds=boundaries)
        else:
            para = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(Y, model),
                                                   updating='deferred', workers=-1)
        alpha, beta = para.x
    else:
        if (alpha < boundaries[0][0] or alpha > boundaries[0][1]) or (beta < boundaries[1][0] or beta > boundaries[1][1]):
            raise Exception('参数初始值超限')
    print('paras', alpha, beta)

    if weights is None:
        weights = []
        for i in range(1, len(Y) + 1):
            weights.append(i)
        weights = np.array(weights) / sum(weights)
        a = [np.average(Y, weights=weights)]
        b = [(sum(Y[int(np.ceil(len(Y) / 2)):]) - sum(Y[:int(np.floor(len(Y) / 2))])) / (np.floor(len(Y) / 2)) ** 2]
        y = [a[0] + b[0]]
    elif weights == 'equal':
        weights = np.array([1 / len(Y)] * len(Y))
        a = [np.average(Y, weights=weights)]
        b = [(sum(Y[int(np.ceil(len(Y) / 2)):]) - sum(Y[:int(np.floor(len(Y) / 2))])) / (np.floor(len(Y) / 2)) ** 2]
        y = [a[0] + b[0]]
    else:
        weights = np.array([1 / len(Y)] * len(Y))
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        y = [a[0] + b[0]]

    for i in range(len(Y) + fc):
        if i == len(Y):
            Y.append(a[-1] + b[-1])
        a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        y.append(a[i + 1] + b[i + 1])

    if sum(np.array(Y[-fc:]) - np.array(y[-fc-1:-1])) != 0:
        print(Y[-fc:], '\n', y[-fc-1:-1])
        raise Exception('预测值Y和y不相等')

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 2)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    return {'predict': y[-fc-1:-1], 'alpha': alpha, 'beta': beta, 'rmse': rmse_index, 'aic': aic_index,
            'std_residual': std_residual, 'fittedvalues': y[:-fc-1], 'weights': weights}


def double_mul(x, fc, alpha=None, beta=None, boundaries = [(1 / 2 * 1e-1, 1 - 1 / 2 * 1e-1)] * 2, method='TNC',
           initial_values = np.array([0.2] * 2), weights=None):
    """
        双参数指数平滑乘法模型
        parameters:
            --------
            x: 时间序列数据列表
            fc: 预测步数
            alpha: level 项 a 的参数,初始值默认为空
            beta: trend 项 b 的参数,初始值默认为空
            boundaries：参数 alpha，beta 训练时的边界值
            initial_values：参数 alpha，beta 训练时使用的初始值，最好使用同序列上一次训练所得参数值，大概率可减小梯度下降的迭代次数
            method：目标函数做梯度下降时的方法，优化算法可选['L-BFGS-B','SLSQP','TNC','trust-constr','global']
            weights: 对输入模型的历史值设置权重，用于给各迭代项的首项赋值；可为None，则越近的点权重越高，越远的点权重越低；可为equal，则等权重；
            或其他任何字符，则无权重，即a[0]=Y[0],b[0]=Y[1]-Y[0]

        return:
            --------
            predict: 一个预测期的预测值
            alpha: level 项的非训练指定参数，或经训练的最优参数
            beta: trend 项的非训练指定参数，或经训练的最优参数
            rmse：计算历史期真实值和拟合值的均方根误差，与小越好
            aic：计算历史期真实值和拟合值的信息准则，判定拟合的优良程度，越小越好
            std_residual: 真实序列与预测序列残差的标准差，越小越好
            fittedvalues： 历史区间的拟合值
    """

    Y = x[:]
    if alpha is None or beta is None:
        model = 'double_mul'
        if method != 'global':
            para = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model), method=method, jac='2-point',
                                            bounds=boundaries)
        else:
            para = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(Y, model),
                                                   updating='deferred', workers=-1)
        alpha, beta = para.x
    else:
        if (alpha < boundaries[0][0] or alpha > boundaries[0][1]) or (beta < boundaries[1][0] or beta > boundaries[1][1]):
            raise Exception('参数初始值超限')
    print('paras', alpha, beta)

    if weights is None:
        weights = []
        for i in range(1, len(Y) + 1):
            weights.append(i)
        weights = np.array(weights) / sum(weights)
        a = [np.average(Y, weights=weights)]
        b = [(sum(Y[int(np.ceil(len(Y) / 2)):]) - sum(Y[:int(np.floor(len(Y) / 2))])) / (np.floor(len(Y) / 2)) ** 2]
        y = [a[0] * b[0]]
    elif weights == 'equal':
        weights = np.array([1 / len(Y)] * len(Y))
        a = [np.average(Y, weights=weights)]
        b = [(sum(Y[int(np.ceil(len(Y) / 2)):]) - sum(Y[:int(np.floor(len(Y) / 2))])) / (np.floor(len(Y) / 2)) ** 2]
        y = [a[0] * b[0]]
    else:
        weights = np.array([1 / len(Y)] * len(Y))
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        y = [a[0] * b[0]]

    for i in range(len(Y) + fc):
        if i == len(Y):
            Y.append(a[-1] * b[-1])
        a.append(alpha * Y[i] + (1 - alpha) * (a[i] * b[i]))
        b.append(beta * (a[i + 1] / a[i]) + (1 - beta) * b[i])
        y.append(a[i + 1] * b[i + 1])

    if sum(np.array(Y[-fc:]) - np.array(y[-fc-1:-1])) != 0:
        print(Y[-fc:], '\n', y[-fc-1:-1])
        raise Exception('预测值Y和y不相等')

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 2)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    return {'predict': y[-fc-1:-1], 'alpha': alpha, 'beta': beta, 'rmse': rmse_index, 'aic': aic_index,
            'std_residual': std_residual, 'fittedvalues': y[:-fc-1], 'weights': weights}


def additive(x, m, fc, alpha=None, beta=None, gamma=None,
             boundaries = [(1 / 2 * 1e-1, 1 - 1 / 2 * 1e-1), (1 / 2 * 1e-1, 1 - 1 / 2 * 1e-1), (1e-1, 1 - 1e-1)],
             method='trust-constr', initial_values = np.array([0.2, 0.2, 2 / 3]), weights=None):
    """
        三参数指数平滑加法模型
        parameters:
            --------
            x: 时间序列数据列表
            m：季节项所用周期，用以反映历史数据的周期性规律
            fc: 预测步数
            alpha: level 项 a 的参数,初始值默认为空
            beta: trend 项 b 的参数,初始值默认为空
            gamma：season 项 s 的参数，初始值默认为空
            boundaries：参数 alpha，beta，gamma 训练时的边界值
            initial_values：参数 alpha，beta，gamma 训练时使用的初始值，最好使用同序列上一次训练所得参数值，大概率可减小梯度下降的迭代次数
            method：目标函数做梯度下降时的方法，优化算法可选['L-BFGS-B', 'SLSQP', 'TNC', 'trust-constr', 'global']
            weights: 对输入模型的历史值设置权重，用于给各迭代项的首项赋值；可为None，则越近的点权重越高，越远的点权重越低；可为equal，则等权重；
            或其他任何字符，则无权重，即a[0]=Y[0], b[0]=Y[1]-Y[0], s=[Y[i] - a[0] for i in range(m)]

        return:
            --------
            predict: 一个预测期的预测值
            alpha: level 项的非训练指定参数，或经训练的最优参数
            beta: trend 项的非训练指定参数，或经训练的最优参数
            gamma: season 项的非训练指定参数，或经训练的最优参数
            rmse：计算历史期真实值和拟合值的均方根误差，与小越好
            aic：计算历史期真实值和拟合值的信息准则，判定拟合的优良程度，越小越好
            std_residual: 真实序列与预测序列残差的标准差，越小越好
            fittedvalues： 历史区间的拟合值
    """

    Y = x[:]
    if alpha is None or beta is None or gamma is None:
        model = 'additive'
        if method != 'global':
            para = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, m), method=method, jac='2-point',
                                     bounds=boundaries)
        else:
            para = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(Y, model, m),
                                                   updating='deferred', workers=-1)
        alpha, beta, gamma = para.x
    else:
        if (alpha < boundaries[0][0] or alpha > boundaries[0][1]) or (
                beta < boundaries[1][0] or beta > boundaries[1][1]) or (
            gamma < boundaries[2][0] or gamma > boundaries[2][1]):
            raise Exception('参数初始值超限')
    print('paras', alpha, beta, gamma)

    if weights is None:
        weights = []
        for i in range(1, len(Y) + 1):
            weights.append(i)
        weights = np.array(weights) / sum(weights)
        a = [np.average(Y, weights=weights)]
        b = [(sum(Y[int(np.ceil(len(Y) / 2)):]) - sum(Y[:int(np.floor(len(Y) / 2))])) / (np.floor(len(Y) / 2)) ** 2]
        if len(Y) > 2*m:
            s = (np.average(np.array(Y[:len(Y) // m * m]).reshape(-1, m), axis=0,
                            weights=[sum(weights[:m])+1, sum(weights[m:2*m])+1]) - a[0]).tolist()
        else:
            s = (np.average(np.array(Y[:len(Y) // m * m]).reshape(-1, m), axis=0) - a[0]).tolist()
        y = [a[0] + b[0] + s[0]]
    elif weights == 'equal':
        weights = np.array([1 / len(Y)] * len(Y))
        a = [np.average(Y, weights=weights)]
        b = [(sum(Y[int(np.ceil(len(Y) / 2)):]) - sum(Y[:int(np.floor(len(Y) / 2))])) / (np.floor(len(Y) / 2)) ** 2]
        if len(Y) > 2*m:
            s = (np.average(np.array(Y[:len(Y) // m * m]).reshape(-1, m), axis=0,
                            weights=[sum(weights[:m])+1, sum(weights[m:2*m])+1]) - a[0]).tolist()
        else:
            s = (np.average(np.array(Y[:len(Y) // m * m]).reshape(-1, m), axis=0) - a[0]).tolist()
        y = [a[0] + b[0] + s[0]]
    else:  # a,b,s的初始值不考虑历史值的全长信息
        weights = np.array([1 / len(Y)] * len(Y))
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        s = [Y[i] - a[0] for i in range(m)]
        y = [a[0] + b[0] + s[0]]

    for i in range(len(Y) + fc):
        if i == len(Y):
            Y.append(round(a[-1] + b[-1] + s[-m], 4))
        a.append(alpha * (Y[i] - s[-m]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] - a[i + 1]) + (1 - gamma) * s[-m])
        y.append(round(a[i + 1] + b[i + 1] + s[-m], 4))

    print(Y[-fc:], '\n', y[-fc:])
    if sum(np.array(Y[-fc:]) - np.array(y[-fc-1:-1])) != 0:
        print(Y[-fc:], y[-fc-1:-1])
        raise Exception('预测值Y和y不相等')

    if sum(np.array(Y[-fc:]) - np.array(y[-fc-1:-1])) != 0:
        print(Y[-fc:], y[-fc-1:-1])
        raise Exception('预测值Y和y不相等')
    if not ((len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s) == m + len(x) + fc)):
        print(s)
        raise Exception('分项长度有误，不能使用季节项序列')

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 3)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    return {'predict': y[-fc-1:-1], 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'rmse': rmse_index, 'aic': aic_index,
            'std_residual': std_residual, 'max(s)': max(s), 'min(s)': min(s), 'mean(s)': np.mean(s), 'seasonal': s,
            'fittedvalues': y[:-fc-1], 'weights': weights}


def multiplicative(x, m, fc, alpha=None, beta=None, gamma=None,
                   boundaries = [(1 / 2 * 1e-1, 1 - 1 / 2 * 1e-1), (1 / 2 * 1e-1, 1 - 1 / 2 * 1e-1), (1e-1, 1 - 1e-1)],
                   method='L-BFGS-B', initial_values = np.array([0.2, 0.2, 2 / 3]), weights=None):
    """
        三参数指数平滑乘法模型
        parameters:
            --------
            x:时间序列数据列表
            m：季节项所用周期，用以反映历史数据的周期性规律
            fc:预测步数
            alpha: level 项 a 的参数,初始值默认为空
            beta: trend 项 b 的参数,初始值默认为空
            gamma：season 项 s 的参数，初始值默认为空
            boundaries：参数 alpha，beta，gamma 训练时的边界值
            initial_values：参数 alpha，beta，gamma 训练时使用的初始值，最好使用同序列上一次训练所得参数值，大概率可减小梯度下降的迭代次数
            method：目标函数做梯度下降时的方法，优化算法可选['L-BFGS-B','SLSQP','TNC','trust-constr','global']
            weights:对输入模型的历史值设置权重，用于给各迭代项的首项赋值；可为None，则越近的点权重越高，越远的点权重越低；可为equal，则等权重；
            或其他任何字符，则无权重，即a[0]=Y[0], b[0]=Y[1]-Y[0], s=[Y[i] / a[0] for i in range(m)]

        return:
            --------
            predict:一个预测期的预测值
            alpha: level 项的非训练指定参数，或经训练的最优参数
            beta: trend 项的非训练指定参数，或经训练的最优参数
            gamma: season 项的非训练指定参数，或经训练的最优参数
            rmse：计算历史期真实值和拟合值的均方根误差，与小越好
            aic：计算历史期真实值和拟合值的信息准则，判定拟合的优良程度，越小越好
            std_residual: 真实序列与预测序列残差的标准差，越小越好
            fittedvalues： 历史区间的拟合值
    """
    Y = x[:]
    # 执行模型训练，得到各分项的参数alpha、beta、gamma；或直接使用指定的参数值。
    if alpha is None or beta is None or gamma is None:
        model = 'multiplicative'
        if method != 'global':
            para = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, m), method=method, jac='2-point',
                                     bounds=boundaries)
        else:
            para = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(Y, model, m),
                                                   updating='deferred', workers=-1)
        alpha, beta, gamma = para.x
    else:
        if (alpha < boundaries[0][0] or alpha > boundaries[0][1]) or (
                beta < boundaries[1][0] or beta > boundaries[1][1]) or (
            gamma < boundaries[2][0] or gamma > boundaries[2][1]):
            raise Exception('参数初始值超限')
    print('paras', alpha, beta, gamma)

    # 设定各分项的启动初始值
    if weights is None:
        weights = []
        for i in range(1, len(Y) + 1):
            weights.append(i)
        weights = np.array(weights) / sum(weights)
        '''给截距a、斜率b、非线性项s、三参数乘法模型的目标函数y赋初值，以启动这四项的递归。
        a,b,s需要赋初值的原因是使基于递归表达式的模型在下面的循环中可以启动，并反映真实值的整体信息。
        应采用真实值在整个训练集上的加权均值(a[0])，斜率的均值(b[0])，加权平均偏离程度s[:m]，可最大程度避免随机波动和离群值的影响。'''
        a = [np.average(Y, weights=weights)]
        '''训练集上前一半真实值和后一半真实值中每对点斜率的均值，len(Y)是奇偶数均可，此时len(Y)只需 >= m 即可'''
        b = [(sum(Y[int(np.ceil(len(Y) / 2)):]) - sum(Y[:int(np.floor(len(Y) / 2))])) / (np.floor(len(Y) / 2)) ** 2]
        '''真实值在整个训练集上所有完整周期的每个对应位置上的均值与整个训练集上真实值均值的比值，len(s)=m，即s的初始序列长度为所选周期长度m'''
        if len(Y) > 2*m:
            s = (np.average(np.array(Y[:len(Y) // m * m]).reshape(-1, m), axis=0,
                            weights=[sum(weights[:m])+1, sum(weights[m:2*m])+1]) / a[0]).tolist()
        else:
            s = (np.average(np.array(Y[:len(Y) // m * m]).reshape(-1, m), axis=0) / a[0]).tolist()
        '''人为给预测序列y赋初值，可使循环后a,b,y等长；此处也可令y=[]，则循环后预测序列y与真实值序列Y等长。即此处给y赋初值不是必要操作，y[0]不能使用。'''
        y = [(a[0] + b[0]) * s[0]]
    elif weights == 'equal':
        weights = np.array([1 / len(Y)] * len(Y))
        a = [np.average(Y, weights=weights)]
        b = [(sum(Y[int(np.ceil(len(Y) / 2)):]) - sum(Y[:int(np.floor(len(Y) / 2))])) / (np.floor(len(Y) / 2)) ** 2]
        if len(Y) > 2*m:
            s = (np.average(np.array(Y[:len(Y) // m * m]).reshape(-1, m), axis=0,
                            weights=[sum(weights[:m])+1, sum(weights[m:2*m])+1]) / a[0]).tolist()
        else:
            s = (np.average(np.array(Y[:len(Y) // m * m]).reshape(-1, m), axis=0) / a[0]).tolist()
        y = [(a[0] + b[0]) * s[0]]
    else:  # a,b,s的初始值不考虑历史值的全长信息
        weights = np.array([1 / len(Y)] * len(Y))
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        s = [Y[i] / a[0] for i in range(m)]
        y = [(a[0] + b[0]) * s[0]]

    '''模型迭代，得到历史期的拟合值y[1:-fc - 1]和预测期的预测值y[-fc-1:-1]'''
    for i in range(len(Y) + fc):
        # s[]的索引中不能含有i-m，否则会出现第一个周期采用负索引，后面的周期采用正索引的情况；导致在正索引的第一个周期中，s本该取到序列中第二个周期的值，但却重复取到第一个周期的值。
        if i == len(Y):
            # 将Y.append()放在各分项的append之前，则输出的是离当期最近的fc个点的预测值；y最后一个点不能用，因为是距当期fc+1个点的预测值。
            # a、b序列的最后一个值不能用，即取到序列的倒数第二个值为止；而Y.append在a,b,s,y.append之前，所以a,b,s的索引为-1,-1,-m。
            Y.append((a[-1] + b[-1]) * s[-m])
        # s序列取到的元素需要恰好比a,Y序列的元素在索引上向前推一个周期。
        a.append(alpha * (Y[i] / s[-m]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        # 右边s序列取到的元素需要恰好比a,Y,s.append后的序列的元素在索引上向前推一个周期。
        s.append(gamma * (Y[i] / a[i + 1]) + (1 - gamma) * s[-m])
        # a[i+1]，b[i+1]代表Y[i]的信息，用来计算y[i+1]；又因为y含有人为赋值但不可用初值y[0]，所以s[-m]正好比y[i+1]向前推一个周期。
        y.append((a[i + 1] + b[i + 1]) * s[-m])

    if sum(np.array(Y[-fc:]) - np.array(y[-fc-1:-1])) != 0:
        print(Y[-fc:], y[-fc-1:-1])
        raise Exception('预测值Y和y不相等')
    if not ((len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s) == m + len(x) + fc)):
        print(s)
        raise Exception('分项长度有误，不能使用季节项序列')

    '''拟合程度指标，只需比较历史期，不能比较预测期上的指标，因为历史期上才有真实销量，预测期上无真实销量'''
    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 3)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    return {'predict': y[-fc-1:-1], 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'rmse': rmse_index, 'aic': aic_index,
            'std_residual': std_residual, 'max(s)': max(s), 'min(s)': min(s), 'mean(s)': np.mean(s), 'seasonal': s,
            'fittedvalues': y[:-fc-1], 'weights': weights}


def double_seasonal_add_add_day(x, m_year=365-7*2, m_week=7, fc=4*7, alpha=None, beta=None, gamma_year=None, gamma_week=None,
                           boundaries=[(1 / 2 * 1e-1, 1 - 1 / 2 * 1e-1), (1 / 2 * 1e-1, 1 - 1 / 2 * 1e-1),
                                       (1e-1, 1 - 1e-1), (1e-1, 1 - 1e-1)],
                           method='L-BFGS-B', initial_values=np.array([0.2, 0.2, 2 / 3, 2/3]), weights=None
                           ):
    """
        四参数指数平滑双重季节性加法加法日序列模型
        parameters:
            --------
            x: 时间序列数据列表
            m_year：年季节项所用周期，用以反映历史数据的年周期性规律
            m_week：周季节项所用周期，用以反映历史数据的周周期性规律
            fc: 预测步数
            alpha: level项 a 的参数,初始值默认为空
            beta: trend项 b 的参数,初始值默认为空
            gamma_year：season项 s_year 的参数，初始值默认为空
            gamma_week：season项 s_week 的参数，初始值默认为空
            boundaries：参数 alpha，beta，gamma_year，gamma_week 训练时的边界值
            initial_values：参数 alpha，beta，gamma_year，gamma_week 训练时使用的初始值，最好使用同序列上一次训练所得参数值，大概率可减小梯度下降的迭代次数
            method：目标函数做梯度下降时的方法，优化算法可选['L-BFGS-B', 'SLSQP', 'TNC', 'trust-constr', 'global']
            weights: 对输入模型的历史值设置权重，用于给各迭代项的首项赋值；可为None，则越近的点权重越高，越远的点权重越低；可为equal，则等权重；
            或其他任何字符，则无权重，即 a[0]=Y[0], b[0]=Y[1]-Y[0], s_year = [Y[i] - a[0] for i in range(m_year)],
            s_week = [Y[i] - a[0] for i in range(m_week)]

        return:
            --------
            predict: 一个预测期的预测值
            alpha: level 项的非训练指定参数，或经训练的最优参数
            beta: trend 项的非训练指定参数，或经训练的最优参数
            gamma_year: s_year 项的非训练指定参数，或经训练的最优参数
            gamma_week: s_week 项的非训练指定参数，或经训练的最优参数
            rmse：计算历史期真实值和拟合值的均方根误差，与小越好
            aic：计算历史期真实值和拟合值的信息准则，判定拟合的优良程度，越小越好
            std_residual: 真实序列与预测序列残差的标准差，越小越好
            fittedvalues： 历史区间的拟合值
    """

    Y = x[:]
    if alpha is None or beta is None or gamma_year is None or gamma_week is None:
        model = 'double_seasonal_add_add_day'
        if method != 'global':
            para = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_week)), method=method,
                                     jac='2-point', bounds=boundaries)
        else:
            para = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(Y, model, (m_year, m_week)),
                                                   updating='deferred', workers=-1)
        alpha, beta, gamma_year, gamma_week = para.x
    else:
        if (alpha < boundaries[0][0] or alpha > boundaries[0][1]) or (
                beta < boundaries[1][0] or beta > boundaries[1][1]) or (
            gamma_year < boundaries[2][0] or gamma_year > boundaries[2][1]) or (
                gamma_week < boundaries[3][0] or gamma_week > boundaries[3][1]):
            raise Exception('参数初始值超限')
    print('paras', alpha, beta, gamma_year, gamma_week)

    if weights is None:
        weights = []
        for i in range(1, len(Y) + 1):
            weights.append(i)
        weights = np.array(weights) / sum(weights)
        a = [np.average(Y, weights=weights)]
        b = [(sum(Y[int(np.ceil(len(Y) / 2)):]) - sum(Y[:int(np.floor(len(Y) / 2))])) / (np.floor(len(Y) / 2)) ** 2]
        if len(Y) > 2*m_year:
            s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0,
                            weights=[sum(weights[:m_year])+1, sum(weights[m_year:2*m_year])+1]) - a[0]).tolist()
            s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                 weights=[sum(weights[:m_week]), sum(weights[m_week:2 * m_week])]) - a[0]).tolist()
        else:
            s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) - a[0]).tolist()
            s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                 weights=[sum(weights[:m_week]), sum(weights[m_week:2 * m_week])]) - a[0]).tolist()
        y = [a[0] + b[0] + s_year[0] + s_week[0]]
    elif weights == 'equal':
        weights = np.array([1 / len(Y)] * len(Y))
        a = [np.average(Y, weights=weights)]
        b = [(sum(Y[int(np.ceil(len(Y) / 2)):]) - sum(Y[:int(np.floor(len(Y) / 2))])) / (np.floor(len(Y) / 2)) ** 2]
        if len(Y) > 2*m_year:
            s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0,
                            weights=[sum(weights[:m_year])+1, sum(weights[m_year:2*m_year])+1]) - a[0]).tolist()
            s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                 weights=[sum(weights[:m_week]), sum(weights[m_week:2 * m_week])]) - a[0]).tolist()
        else:
            s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) - a[0]).tolist()
            s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                 weights=[sum(weights[:m_week]), sum(weights[m_week:2 * m_week])]) - a[0]).tolist()
        y = [a[0] + b[0] + s_year[0] + s_week[0]]
    else:  # 各分项的初始值不考虑历史值的全长信息；当输入模型的数据长度越短，例如不足2个m_year时，对应周期s_year项初始值的设定就越重要；
        # 当输入模型的数据长度越长，例如大于2个周期长度m_year甚至越多时，对应周期s_year项初始值的不同设定就越不重要。
        weights = np.array([1 / len(Y)] * len(Y))
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        s_year = [Y[i] - a[0] for i in range(m_year)]
        s_week = [Y[i] - a[0] for i in range(m_week)]
        y = [a[0] + b[0] + s_year[0] + s_week[0]]

    for i in range(len(Y) + fc):
        if i == len(Y):
            Y.append((a[-1] + b[-1]) + s_year[-m_year] + s_week[-m_week])
        a.append(alpha * (Y[i] - (s_year[-m_year] + s_week[-m_week])) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s_year.append(gamma_year * (Y[i] - (a[i + 1] + s_week[-m_week])) + (1 - gamma_year) * s_year[-m_year])
        s_week.append(gamma_week * (Y[i] - (a[i + 1] + s_year[-m_year])) + (1 - gamma_week) * s_week[-m_week])
        y.append((a[i + 1] + b[i + 1]) + s_year[-m_year] + s_week[-m_week])

    if sum(np.array(Y[-fc:]) - np.array(y[-fc-1:-1])) != 0:
        print(Y[-fc:], y[-fc-1:-1])
        raise Exception('预测值Y和y不相等')
    if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (
            len(s_week) == m_week + len(x) + fc):
        s_total = (np.array(s_year[m_year - m_week:]) + np.array(s_week[:])).tolist()
    else:
        print(s_total)
        raise Exception('分项长度有误，不能使用季节项序列')

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 4)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    return {'len(s_year)': len(s_year), 'max(s_year)': max(s_year), 'min(s_year)': min(s_year),
            'mean(s_year)': np.mean(s_year), 'np.std(s_year)': np.std(s_year), 's_year_75': np.percentile(s_year, 75),
            's_year_25': np.percentile(s_year, 25), 's_year_50': np.percentile(s_year, 50), 's_year': s_year,
            'len(s_week)': len(s_week), 'max(s_week)': max(s_week), 'min(s_week)': min(s_week),
            'mean(s_week)': np.mean(s_week), 'np.std(s_week)': np.std(s_week), 's_week_75': np.percentile(s_week, 75),
            's_week_25': np.percentile(s_week, 25), 's_week_50': np.percentile(s_week, 50), 's_week': s_week,
            'res': rmse_index - std_residual,
            'predict': y[-fc-1:-1], 'alpha': alpha, 'beta': beta,
            'gamma_year': gamma_year, 'gamma_week': gamma_week,
            'rmse': rmse_index, 'aic': aic_index, 'std_of_residual': std_residual,
            'len(s_total)': len(s_total), 'max(s_total)': max(s_total), 'min(s_total)': min(s_total),
            'mean(s_total)': np.mean(s_total), 'np.std(s_total)': np.std(s_total), 's_total_75': np.percentile(s_total, 75),
            's_total_25': np.percentile(s_total, 25), 's_total_50': np.percentile(s_total, 50),
            's_total': s_total, 'fittedvalues': y[:-fc-1], 'weights': weights}


def multi_seasonal_add_day(x, m_year=365, m_quarter=round(365 / 4), m_month=round(365 / 12), m_week=7, fc=4 * 7,
                          alpha=None, beta=None, gamma_year=None, gamma_quarter=None, gamma_month=None,
                          gamma_week=None):
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
        initial_values = np.array([.2] * (6 - 4) + [2 / 3] * 4)
        boundaries = [(5*1e-2, 1 - 5*1e-2)] * (len(initial_values) - 4) + [(1e-1, 1 - 1e-1)] * 4
        model = 'multi_seasonal_add_day'
        # para_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month, m_week)),
        #                                method='L-BFGS-B', jac='2-point', bounds=boundaries)
        # alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = para_slsqp.x

        para_rmse_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month, m_week)), method='L-BFGS-B', jac='2-point',
                                            bounds=boundaries)
        # para_ols_slsqp = optimize.minimize(ols_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month, m_week)), method='L-BFGS-B', jac='2-point',
        #                                    bounds=boundaries)
        # para_differential_evolution = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(
        #     Y, model, (m_year, m_quarter, m_month, m_week)), updating='deferred', workers=-1)
        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = para_rmse_slsqp.x
        # alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = para_differential_evolution.x
        # print('rmse_loss_func', para_rmse_slsqp.x)
        # print('ols_loss_func', para_ols_slsqp.x)
        print('para_rmse_slsqp', para_rmse_slsqp.x)

    a = [sum(Y[:]) / float(len(Y))]
    b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
    s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
    s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[0]).tolist()
    s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
    s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) - a[0]).tolist()
    y = [(a[0] + b[0]) + s_year[0] + s_quarter[0] + s_month[0] + s_week[0]]

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
        y.append((a[i + 1] + b[i + 1]) + s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month] + s_week[-m_week])

    if sum(np.array(Y[-fc:]) - np.array(y[-fc-1:-1])) != 0:
        print(Y[-fc:], y[-fc-1:-1])
        raise Exception('预测值Y和y不相等')
    if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (
            len(s_quarter) == m_quarter + len(x) + fc) and (len(s_month) == m_month + len(x) + fc) and (
            len(s_week) == m_week + len(x) + fc):
        s_total = (np.array(s_year[m_year - m_week:]) + np.array(s_quarter[m_quarter - m_week:]) + np.array(
            s_month[m_month - m_week:]) + np.array(s_week[:])).tolist()
    else:
        s_total = []
        print('分项长度有误，不能使用季节项序列')

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 6)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    # return {'predict': Y[-fc:], 'alpha': alpha, 'beta': beta, 'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
    #         'gamma_month': gamma_month, 'gamma_week': gamma_week, 'rmse': rmse_index, 'aic': aic_index,
    #         'std_residual': std_residual, 'max(s)': max(s_total),
    #         'min(s)': min(s_total), 'seasonal': s_total}
    # return {'len(s_year)': len(s_year), 'max(s_year)': max(s_year), 'min(s_year)': min(s_year),
    #         'mean(s_year)': np.mean(s_year), 'np.std(s_year)': np.std(s_year), 's_year75': np.percentile(s_year, 75),
    #         's_year25': np.percentile(s_year, 25),
    #         'len(s_quarter)': len(s_quarter), 'max(s_quarter)': max(s_quarter), 'min(s_quarter)': min(s_quarter),
    #         'mean(s_quarter)': np.mean(s_quarter), 'np.std(s_quarter)': np.std(s_quarter),
    #         's_quarter75': np.percentile(s_quarter, 75),
    #         's_quarter25': np.percentile(s_quarter, 25),
    #         'len(s_month)': len(s_month), 'max(s_month)': max(s_month), 'min(s_month)': min(s_month),
    #         'mean(s_month)': np.mean(s_month), 'np.std(s_month)': np.std(s_month),
    #         's_month75': np.percentile(s_month, 75),
    #         's_month25': np.percentile(s_month, 25),
    #         'len(s_week)': len(s_week), 'max(s_week)': max(s_week), 'min(s_week)': min(s_week),
    #         'mean(s_week)': np.mean(s_week), 'np.std(s_week)': np.std(s_week),
    #         's_week75': np.percentile(s_week, 75),
    #         's_week25': np.percentile(s_week, 25),
    #         'res': rmse_index - std_residual,
    #         'predict': Y[-fc:], 'alpha': alpha, 'beta': beta,
    #         'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
    #         'gamma_month': gamma_month, 'gamma_week': gamma_week,
    #         'rmse': rmse_index, 'aic': aic_index, 'std_residual': std_residual,
    #         'max(s_total)': max(s_total), 'min(s_total)': min(s_total), 'mean(s_total)': np.mean(s_total),
    #         's_total': s_total, 'fittedvalues': y[:-fc-1]}
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
            'predict': y[-fc-1:-1], 'alpha': alpha, 'beta': beta,
            'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
            'gamma_month': gamma_month, 'gamma_week': gamma_week,
            'rmse': rmse_index, 'aic': aic_index, 'std_residual': std_residual,
            'max(s_total)': max(s_total), 'min(s_total)': min(s_total), 'mean(s_total)': np.mean(s_total),
            's_total': s_total, 'fittedvalues': y[:-fc-1]}


def multi_seasonal_add_week(x, m_year=52, m_quarter=round(52 / 4), m_month=round(52 / 12), fc=4,
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
        initial_values = np.array([.2] * (5 - 3) + [2 / 3] * 3)
        boundaries = [(5*1e-2, 1 - 5*1e-2)] * (len(initial_values) - 3) + [(1e-1, 1 - 1e-1)] * 3
        model = 'multi_seasonal_add_week'
        # para_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month)),
        #                                method='L-BFGS-B', jac='2-point', bounds=boundaries)
        # alpha, beta, gamma_year, gamma_quarter, gamma_month = para_slsqp.x

        para_rmse_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month)), method='L-BFGS-B', jac='2-point',
                                            bounds=boundaries)
        # para_ols_slsqp = optimize.minimize(ols_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month)), method='L-BFGS-B', jac='2-point',
        #                                    bounds=boundaries)
        # para_differential_evolution = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(
        #     Y, model, (m_year, m_quarter, m_month)), updating='deferred', workers=-1)
        alpha, beta, gamma_year, gamma_quarter, gamma_month = para_rmse_slsqp.x
        # alpha, beta, gamma_year, gamma_quarter, gamma_month = para_differential_evolution.x
        # print('rmse_loss_func', para_rmse_slsqp.x)
        # print('ols_loss_func', para_ols_slsqp.x)
        print('para_rmse_slsqp', para_rmse_slsqp.x)

    a = [sum(Y[:]) / float(len(Y))]
    b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
    s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) - a[0]).tolist()
    s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) - a[0]).tolist()
    s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) - a[0]).tolist()
    y = [(a[0] + b[0]) + s_year[0] + s_quarter[0] + s_month[0]]

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
        y.append((a[i + 1] + b[i + 1]) + s_year[-m_year] + s_quarter[-m_quarter] + s_month[-m_month])

    if sum(np.array(Y[-fc:]) - np.array(y[-fc-1:-1])) != 0:
        print(Y[-fc:], y[-fc-1:-1])
        raise Exception('预测值Y和y不相等')
    if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (
            len(s_quarter) == m_quarter + len(x) + fc) and (len(s_month) == m_month + len(x) + fc):
        s_total = (np.array(s_year[m_year - m_month:]) + np.array(s_quarter[m_quarter - m_month:]) + np.array(
            s_month[:])).tolist()
    else:
        s_total = []
        print('分项长度有误，不能使用季节项序列')

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 5)
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
            's_total': s_total, 'fittedvalues': y[:-fc-1]}


def multi_seasonal_mul_day(x, m_year=365, m_quarter=round(365 / 4), m_month=round(365 / 12), m_week=7, fc=4 * 7,
                          alpha=None, beta=None, gamma_year=None, gamma_quarter=None, gamma_month=None,
                          gamma_week=None):
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
        initial_values = np.array([.2] * 6)
        boundaries = [(1 / 2 * 1e-1, 1 - 1 / 2 * 1e-1)] * 6
        model = 'multi_seasonal_mul_day'
        # para_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month, m_week)), method='L-BFGS-B', jac='2-point',
        #                                bounds=boundaries)
        para_rmse_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month, m_week)), method='L-BFGS-B', jac='2-point',
                                            bounds=boundaries)
        # para_differential_evolution_rmse = optimize.differential_evolution(rmse_loss_func, bounds=boundaries, args=(
        #                                                                     Y, model, (m_year, m_quarter, m_month, m_week)), updating='deferred', workers=-1)
        # para_ols_slsqp = optimize.minimize(ols_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month, m_week)), method='L-BFGS-B', jac='2-point',
        #                                    bounds=boundaries)
        # para_differential_evolution = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(
        # Y, model, (m_year, m_quarter, m_month, m_week)), updating='deferred', workers=-1)
        # para_shgo = optimize.shgo(ols_loss_func, bounds=boundaries, args=(Y, model, (m_year, m_quarter, m_month, m_week)), iters=3)
        alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = para_rmse_slsqp.x
        # alpha, beta, gamma_year, gamma_quarter, gamma_month, gamma_week = para_differential_evolution.x
        # print('rmse_loss_func', para_rmse_slsqp.x)
        # print('ols_loss_func', para_ols_slsqp.x)
        # print('differential_evolution-rmse', para_differential_evolution_rmse.x)
        print('para_rmse_slsqp', para_rmse_slsqp.x)
        # print('shgo-ols', para_shgo.x)

    a = [sum(Y[:]) / float(len(Y))]
    b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
    s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
    s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[0]).tolist()
    s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
    s_week = (np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week).mean(axis=0) / a[0]).tolist()
    y = [(a[0] + b[0]) * s_year[0] * s_quarter[0] * s_month[0] * s_week[0]]

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
        y.append((a[i + 1] + b[i + 1]) * s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month] * s_week[-m_week])

    if sum(np.array(Y[-fc:]) - np.array(y[-fc-1:-1])) != 0:
        print(Y[-fc:], y[-fc-1:-1])
        raise Exception('预测值Y和y不相等')
    if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (
            len(s_quarter) == m_quarter + len(x) + fc) and (len(s_month) == m_month + len(x) + fc) and (
            len(s_week) == m_week + len(x) + fc):
        s_total = (np.array(s_year[m_year - m_week:]) * np.array(s_quarter[m_quarter - m_week:]) * np.array(
            s_month[m_month - m_week:]) * np.array(s_week[:])).tolist()
    else:
        s_total = []
        print('分项长度有误，不能使用季节项序列')

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 6)
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    # return {'predict': Y[-fc:], 'alpha': alpha, 'beta': beta, 'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
    #         'gamma_month': gamma_month, 'gamma_week': gamma_week, 'rmse': rmse_index, 'aic': aic_index,
    #         'std_residual': std_residual, 'max(s)': max(s_total),
    #         'min(s)': min(s_total),
    #         'seasonal': s_total}
    # return {'len(s_year)': len(s_year), 'max(s_year)': max(s_year), 'min(s_year)': min(s_year),
    #         'mean(s_year)': np.mean(s_year), 'np.std(s_year)': np.std(s_year), 's_year75': np.percentile(s_year, 75),
    #         's_year25': np.percentile(s_year, 25),
    #         'len(s_quarter)': len(s_quarter), 'max(s_quarter)': max(s_quarter), 'min(s_quarter)': min(s_quarter),
    #         'mean(s_quarter)': np.mean(s_quarter), 'np.std(s_quarter)': np.std(s_quarter),
    #         's_quarter75': np.percentile(s_quarter, 75),
    #         's_quarter25': np.percentile(s_quarter, 25),
    #         'len(s_month)': len(s_month), 'max(s_month)': max(s_month), 'min(s_month)': min(s_month),
    #         'mean(s_month)': np.mean(s_month), 'np.std(s_month)': np.std(s_month),
    #         's_month75': np.percentile(s_month, 75),
    #         's_month25': np.percentile(s_month, 25),
    #         'len(s_week)': len(s_week), 'max(s_week)': max(s_week), 'min(s_week)': min(s_week),
    #         'mean(s_week)': np.mean(s_week), 'np.std(s_week)': np.std(s_week),
    #         's_week75': np.percentile(s_week, 75),
    #         's_week25': np.percentile(s_week, 25),
    #         'res': rmse_index - std_residual,
    #         'predict': Y[-fc:], 'alpha': alpha, 'beta': beta,
    #         'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
    #         'gamma_month': gamma_month, 'gamma_week': gamma_week,
    #         'rmse': rmse_index, 'aic': aic_index, 'std_residual': std_residual,
    #         'max(s_total)': max(s_total), 'min(s_total)': min(s_total), 'mean(s_total)': np.mean(s_total),
    #         's_total': s_total, 'fittedvalues': y[:-fc-1]}
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
            'predict': y[-fc-1:-1], 'alpha': alpha, 'beta': beta,
            'gamma_year': gamma_year, 'gamma_quarter': gamma_quarter,
            'gamma_month': gamma_month, 'gamma_week': gamma_week,
            'rmse': rmse_index, 'aic': aic_index, 'std_residual': std_residual,
            'max(s_total)': max(s_total), 'min(s_total)': min(s_total), 'mean(s_total)': np.mean(s_total),
            's_total': s_total, 'fittedvalues': y[:-fc-1]}


def multi_seasonal_mul_week(x, m_year=52, m_quarter=round(52 / 4), m_month=round(52 / 12), fc=4,
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
        initial_values = np.array([.2] * (5 - 3) + [2 / 3] * 3)
        boundaries = [(5*1e-2, 1 - 5*1e-2)] * (len(initial_values) - 3) + [(1e-1, 1 - 1e-1)] * 3
        model = 'multi_seasonal_mul_week'

        para_rmse_slsqp = optimize.minimize(rmse_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month)),
                                            method='L-BFGS-B', jac='2-point',
                                            bounds=boundaries)
        # para_ols_slsqp = optimize.minimize(ols_loss_func, initial_values, args=(Y, model, (m_year, m_quarter, m_month)),
        #                                    method='L-BFGS-B', jac='2-point',
        #                                    bounds=boundaries)
        # para_differential_evolution = optimize.differential_evolution(ols_loss_func, bounds=boundaries, args=(
        #     Y, model, (m_year, m_quarter, m_month)), updating='deferred', workers=-1)
        alpha, beta, gamma_year, gamma_quarter, gamma_month = para_rmse_slsqp.x
        # alpha, beta, gamma_year, gamma_quarter, gamma_month = para_differential_evolution.x
        # print('rmse_loss_func', para_rmse_slsqp.x)
        # print('ols_loss_func', para_ols_slsqp.x)
        print('para_rmse_slsqp', para_rmse_slsqp.x)

    a = [sum(Y[:]) / float(len(Y))]
    b = [(sum(Y[math.ceil(len(Y) / 2):]) - sum(Y[:math.floor(len(Y) / 2)])) / (math.floor(len(Y) / 2)) ** 2]
    s_year = (np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year).mean(axis=0) / a[0]).tolist()
    s_quarter = (np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter).mean(axis=0) / a[0]).tolist()
    s_month = (np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month).mean(axis=0) / a[0]).tolist()
    y = [(a[0] + b[0]) * s_year[0] * s_quarter[0] * s_month[0]]

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
        y.append((a[i + 1] + b[i + 1]) * s_year[-m_year] * s_quarter[-m_quarter] * s_month[-m_month])

    if sum(np.array(Y[-fc:]) - np.array(y[-fc-1:-1])) != 0:
        print(Y[-fc:], y[-fc-1:-1])
        raise Exception('预测值Y和y不相等')
    if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (
            len(s_quarter) == m_quarter + len(x) + fc) and (len(s_month) == m_month + len(x) + fc):
        s_total = (np.array(s_year[m_year - m_month:]) * np.array(s_quarter[m_quarter - m_month:]) * np.array(
            s_month[:])).tolist()
    else:
        s_total = []
        print('分项长度有误，不能使用季节项序列')

    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], 5)
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
            's_total': s_total, 'fittedvalues': y[:-fc-1]}


def ExponentialSmoothingModels(x, granularity='day', m_year=365-7*1, m_week=7, m_month=365/12, m_quarter=365/4, fc=4*7,
                               alpha=None, beta=None, gamma_year=None, gamma_week=None, gamma_month=None, gamma_quarter=None,
                               boundaries=[(1 / 2 * 1e-1, 1 - 1 / 2 * 1e-1), (1 / 2 * 1e-1, 1 - 1 / 2 * 1e-1),
                                       (1e-1, 1 - 1e-1), (1e-1, 1 - 1e-1), (1e-1, 1 - 1e-1), (1e-1, 1 - 1e-1)],
                               model='multi_seasonal_AAAA', method='L-BFGS-B', initial_values=np.array([0.2, 0.2, 2 / 3, 2/3, 2/3, 2/3]),
                               weights=None, type='RMSE'):
    """
    指数平滑模型综合函数，但不包括每个模型的目标函数

    :param x: 输入模型的历史时序列表
    :param granularity: 输入时序的粒度，day or week
    :param m_year: 年季节项所用周期，用以反映历史数据的年周期性规律
    :param m_week: 周季节项所用周期，用以反映历史数据的周周期性规律
    :param m_month: 月季节项所用周期，用以反映历史数据的月周期性规律
    :param m_quarter: 季季节项所用周期，用以反映历史数据的季周期性规律
    :param fc: 预测步数
    :param alpha: level项 a 的参数,初始值默认为空
    :param beta: trend项 b 的参数,初始值默认为空
    :param gamma_year: season项 s_year 的参数，初始值默认为空
    :param gamma_week: season项 s_week 的参数，初始值默认为空
    :param gamma_month: season项 s_month 的参数，初始值默认为空
    :param gamma_quarter: season项 s_quarter 的参数，初始值默认为空
    :param boundaries: 参数 alpha，beta，gamma_year，gamma_week，gamma_month，gamma_quarter 训练时的边界值
    :param model: one of ['simple', 'linear', 'double_mul', 'additive', 'multiplicative', 'double_seasonal_add_add',
           'double_seasonal_add_mul', 'double_seasonal_mul_add', 'double_seasonal_mul_mul',
           'multi_seasonal_AAAA', 'multi_seasonal_AMAA', 'multi_seasonal_MAAA', 'multi_seasonal_MMAA']
    :param method: 目标函数做梯度下降时的优化算法，one of ['L-BFGS-B', 'SLSQP', 'TNC', 'trust-constr', 'global']
    :param initial_values: 参数 alpha，beta，gamma_year，gamma_week, gamma_month, gamma_quarter 训练时使用的初始值，
           最好使用同序列上一次训练所得参数值，大概率可减小梯度下降的迭代次数
    :param weights: 对输入模型的历史值设置权重，用于给各迭代项的首项赋值；可为 None，则越近的点权重越高，越远的点权重越低；
           可为 equal，则等权重；或其他任何字符，则无权重。
           当输入模型的数据长度越短，例如不足2个m_year时，对应周期s_year项初始值的不同设定，对结果的影响差别就越大，选择合适的方法就越重要；
           当输入模型的数据长度越长，例如大于2个周期长度m_year甚至越多时，对应周期s_year项初始值的不同设定就越不重要。
    :param type: loss function type, either 'RMSE' or 'OLS'
    :return: predict: 一个预测期的预测值
             alpha: level 项的非训练指定参数，或经训练的最优参数
             beta: trend 项的非训练指定参数，或经训练的最优参数
             gamma_year: s_year 项的非训练指定参数，或经训练的最优参数
             gamma_week: s_week 项的非训练指定参数，或经训练的最优参数
             s_total: 总季节项，可用于分层时间序列建模
             rmse：计算历史期真实值和拟合值的均方根误差，与小越好
             aic：计算历史期真实值和拟合值的信息准则，判定拟合的优良程度，越小越好
             std_residual: 真实序列与预测序列残差的标准差，越小越好
             fittedvalues： 历史区间的拟合值
    """

    Y = x[:]
    if granularity == 'week' and model in ['multi_seasonal_AAAA', 'multi_seasonal_AMAA', 'multi_seasonal_MAAA',
                                           'multi_seasonal_MMAA']:
        raise Exception('if granularity is week, model cannot be set as one of \'multi_seasonal_AAAA\', '
                        ' \'multi_seasonal_AMAA\', \'multi_seasonal_MAAA\',  \'multi_seasonal_MMAA\', '
                        'i.e. cannot choose multi-seasonal, at least double seasonal, or single seasonal, no seasonal.')
    if alpha is None or beta is None or gamma_year is None or gamma_week is None or gamma_month is None or gamma_quarter is None:
        if method != 'global':
            para = optimize.minimize(loss_func, initial_values, args=(Y, model, type, (m_year, m_week, m_month, m_quarter)),
                                     method=method, jac='2-point', bounds=boundaries)
        else:
            para = optimize.differential_evolution(loss_func, bounds=boundaries, args=(Y, model, type, (m_year, m_week, m_month, m_quarter)),
                                                   updating='deferred', workers=-1)
        # alpha, beta, gamma_year, gamma_week = para.x
    else:
        if (alpha < boundaries[0][0] or alpha > boundaries[0][1]) or (
            beta < boundaries[1][0] or beta > boundaries[1][1]) or (
            gamma_year < boundaries[2][0] or gamma_year > boundaries[2][1]) or (
            gamma_week < boundaries[3][0] or gamma_week > boundaries[3][1]) or (
            gamma_month < boundaries[4][0] or gamma_month > boundaries[4][1]) or (
            gamma_quarter < boundaries[5][0] or gamma_quarter > boundaries[5][1]):
            raise Exception('参数初始值超限')
    print('paras', para.x, len(para.x))

    if weights is None:
        weights = []
        for i in range(1, len(Y) + 1):
            weights.append(i)
        weights = np.array(weights) / sum(weights)
        a = [np.average(Y, weights=weights)]
        b = [(sum(Y[int(np.ceil(len(Y) / 2)):]) - sum(Y[:int(np.floor(len(Y) / 2))])) / (np.floor(len(Y) / 2)) ** 2]
        if model in ['additive', 'double_seasonal_add_add', 'double_seasonal_add_mul', 'multi_seasonal_AAAA',
                     'multi_seasonal_AMAA']:
            if len(Y) >= 2*m_year:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0,
                                weights=[sum(weights[:m_year])+1, sum(weights[m_year:2*m_year])+1]) - a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) - a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0,
                                weights=[sum(weights[:m_month])+1/12, sum(weights[m_month:2*m_month])+1/12]) - a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0,
                                     weights=[sum(weights[:m_quarter])+1/4, sum(weights[m_quarter:2 * m_quarter])+1/4]) - a[0]).tolist()
            elif 2*m_quarter <= len(Y) < 2*m_year:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) - a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) - a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0,
                                weights=[sum(weights[:m_month])+1/12, sum(weights[m_month:2*m_month])+1/12]) - a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0,
                                     weights=[sum(weights[:m_quarter])+1/4, sum(weights[m_quarter:2 * m_quarter])+1/4]) - a[0]).tolist()
            elif 2*m_month <= len(Y) < 2*m_quarter:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) - a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) - a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0,
                                weights=[sum(weights[:m_month])+1/12, sum(weights[m_month:2*m_month])+1/12]) - a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0) - a[0]).tolist()
            elif 2*m_week <= len(Y) < 2*m_month:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) - a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) - a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0) - a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0) - a[0]).tolist()
            else:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) - a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0) - a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0) - a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0) - a[0]).tolist()
        elif model in ['multiplicative', 'double_seasonal_mul_add', 'double_seasonal_mul_mul', 'multi_seasonal_MAAA',
                       'multi_seasonal_MMAA']:
            if len(Y) >= 2*m_year:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0,
                                weights=[sum(weights[:m_year])+1, sum(weights[m_year:2*m_year])+1]) / a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) / a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0,
                                weights=[sum(weights[:m_month])+1/12, sum(weights[m_month:2*m_month])+1/12]) / a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0,
                                     weights=[sum(weights[:m_quarter])+1/4, sum(weights[m_quarter:2 * m_quarter])+1/4]) / a[0]).tolist()
            elif 2*m_quarter <= len(Y) < 2*m_year:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) / a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) / a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0,
                                weights=[sum(weights[:m_month])+1/12, sum(weights[m_month:2*m_month])+1/12]) / a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0,
                                     weights=[sum(weights[:m_quarter])+1/4, sum(weights[m_quarter:2 * m_quarter])+1/4]) / a[0]).tolist()
            elif 2*m_month <= len(Y) < 2*m_quarter:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) / a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) / a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0,
                                weights=[sum(weights[:m_month])+1/12, sum(weights[m_month:2*m_month])+1/12]) / a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0) / a[0]).tolist()
            elif 2*m_week <= len(Y) < 2*m_month:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) / a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) / a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0) / a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0) / a[0]).tolist()
            else:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) / a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0) / a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0) / a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0) / a[0]).tolist()
        else:  # 给季节性分项赋任意值，使最后return格式可以统一
            s_year = [0]
            s_week = [0]
            s_month = [0]
            s_quarter = [0]
            s_total = [0]
    elif weights == 'equal':
        weights = np.array([1 / len(Y)] * len(Y))
        a = [np.average(Y, weights=weights)]
        b = [(sum(Y[int(np.ceil(len(Y) / 2)):]) - sum(Y[:int(np.floor(len(Y) / 2))])) / (np.floor(len(Y) / 2)) ** 2]
        if model in ['additive', 'double_seasonal_add_add', 'double_seasonal_add_mul', 'multi_seasonal_AAAA',
                     'multi_seasonal_AMAA']:
            if len(Y) >= 2*m_year:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0,
                                weights=[sum(weights[:m_year])+1, sum(weights[m_year:2*m_year])+1]) - a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) - a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0,
                                weights=[sum(weights[:m_month])+1/12, sum(weights[m_month:2*m_month])+1/12]) - a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0,
                                     weights=[sum(weights[:m_quarter])+1/4, sum(weights[m_quarter:2 * m_quarter])+1/4]) - a[0]).tolist()
            elif 2*m_quarter <= len(Y) < 2*m_year:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) - a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) - a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0,
                                weights=[sum(weights[:m_month])+1/12, sum(weights[m_month:2*m_month])+1/12]) - a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0,
                                     weights=[sum(weights[:m_quarter])+1/4, sum(weights[m_quarter:2 * m_quarter])+1/4]) - a[0]).tolist()
            elif 2*m_month <= len(Y) < 2*m_quarter:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) - a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) - a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0,
                                weights=[sum(weights[:m_month])+1/12, sum(weights[m_month:2*m_month])+1/12]) - a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0) - a[0]).tolist()
            elif 2*m_week <= len(Y) < 2*m_month:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) - a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) - a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0) - a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0) - a[0]).tolist()
            else:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) - a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0) - a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0) - a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0) - a[0]).tolist()
        elif model in ['multiplicative', 'double_seasonal_mul_add', 'double_seasonal_mul_mul', 'multi_seasonal_MAAA',
                       'multi_seasonal_MMAA']:
            if len(Y) >= 2*m_year:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0,
                                weights=[sum(weights[:m_year])+1, sum(weights[m_year:2*m_year])+1]) / a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) / a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0,
                                weights=[sum(weights[:m_month])+1/12, sum(weights[m_month:2*m_month])+1/12]) / a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0,
                                     weights=[sum(weights[:m_quarter])+1/4, sum(weights[m_quarter:2 * m_quarter])+1/4]) / a[0]).tolist()
            elif 2*m_quarter <= len(Y) < 2*m_year:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) / a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) / a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0,
                                weights=[sum(weights[:m_month])+1/12, sum(weights[m_month:2*m_month])+1/12]) / a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0,
                                     weights=[sum(weights[:m_quarter])+1/4, sum(weights[m_quarter:2 * m_quarter])+1/4]) / a[0]).tolist()
            elif 2*m_month <= len(Y) < 2*m_quarter:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) / a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) / a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0,
                                weights=[sum(weights[:m_month])+1/12, sum(weights[m_month:2*m_month])+1/12]) / a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0) / a[0]).tolist()
            elif 2*m_week <= len(Y) < 2*m_month:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) / a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0,
                                     weights=[sum(weights[:m_week])+1/52, sum(weights[m_week:2 * m_week])+1/52]) / a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0) / a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0) / a[0]).tolist()
            else:
                s_year = (np.average(np.array(Y[:len(Y) // m_year * m_year]).reshape(-1, m_year), axis=0) / a[0]).tolist()
                s_week = (np.average(np.array(Y[:len(Y) // m_week * m_week]).reshape(-1, m_week), axis=0) / a[0]).tolist()
                s_month = (np.average(np.array(Y[:len(Y) // m_month * m_month]).reshape(-1, m_month), axis=0) / a[0]).tolist()
                s_quarter = (np.average(np.array(Y[:len(Y) // m_quarter * m_quarter]).reshape(-1, m_quarter), axis=0) / a[0]).tolist()
        else:
            s_year = [0]
            s_week = [0]
            s_month = [0]
            s_quarter = [0]
            s_total = [0]
    else:  # 各分项的初始值不考虑历史值的全长信息
        # weights = np.array([1 / len(Y)] * len(Y))
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        if model in ['additive', 'double_seasonal_add_add', 'double_seasonal_add_mul', 'multi_seasonal_AAAA',
                     'multi_seasonal_AMAA']:
            s_year = [Y[i] - a[0] for i in range(m_year)]
            s_week = [Y[i] - a[0] for i in range(m_week)]
            s_month = [Y[i] - a[0] for i in range(m_month)]
            s_quarter = [Y[i] - a[0] for i in range(m_quarter)]
        elif model in ['multiplicative', 'double_seasonal_mul_add', 'double_seasonal_mul_mul', 'multi_seasonal_MAAA',
                       'multi_seasonal_MMAA']:
            s_year = [Y[i] / a[0] for i in range(m_year)]
            s_week = [Y[i] / a[0] for i in range(m_week)]
            s_month = [Y[i] / a[0] for i in range(m_month)]
            s_quarter = [Y[i] / a[0] for i in range(m_quarter)]
        else:
            s_year = [0]
            s_week = [0]
            s_month = [0]
            s_quarter = [0]
            s_total = [0]
    y = [1]  # 给y任意赋一个初值，因为不会使用；只占索引，为了与其他分项的索引保持一致

    if model == 'simple':
        for i in range(len(Y) + fc):
            if i == len(Y):
                Y.append(a[-1])
            a.append(para.x[0] * Y[i] + (1 - para.x[0]) * a[i])
            y.append(a[i + 1])
    elif model == 'linear':
        for i in range(len(Y) + fc):
            if i == len(Y):
                Y.append(a[-1] + b[-1])
            a.append(para.x[0] * Y[i] + (1 - para.x[0]) * (a[i] + b[i]))
            b.append(para.x[1] * (a[i + 1] - a[i]) + (1 - para.x[1]) * b[i])
            y.append(a[i + 1] + b[i + 1])
    elif model == 'double_mul':
        for i in range(len(Y) + fc):
            if i == len(Y):
                Y.append(a[-1] * b[-1])
            a.append(para.x[0] * Y[i] + (1 - para.x[0]) * (a[i] * b[i]))
            b.append(para.x[1] * (a[i + 1] / a[i]) + (1 - para.x[1]) * b[i])
            y.append(a[i + 1] * b[i + 1])
    elif model == 'additive':
        for i in range(len(Y) + fc):
            if i == len(Y):
                Y.append(round(a[-1] + b[-1] + s[-m], 4))
            a.append(para.x[0] * (Y[i] - s[-m]) + (1 - para.x[0]) * (a[i] + b[i]))
            b.append(para.x[1] * (a[i + 1] - a[i]) + (1 - para.x[1]) * b[i])
            s.append(para.x[2] * (Y[i] - a[i + 1]) + (1 - para.x[2]) * s[-m])
            y.append(round(a[i + 1] + b[i + 1] + s[-m], 4))
        if not ((len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s) == m + len(x) + fc)):
            print(s)
            raise Exception('分项长度有误，不能使用季节项序列')
    elif model == 'multiplicative':
        '''模型迭代，得到历史期的拟合值y[1:-fc - 1]和预测期的预测值y[-fc-1:-1]'''
        for i in range(len(Y) + fc):
            # s[]的索引中不能含有i-m，否则会出现第一个周期采用负索引，后面的周期采用正索引的情况；导致在正索引的第一个周期中，s本该取到序列中第二个周期的值，但却重复取到第一个周期的值。
            if i == len(Y):
                # 将Y.append()放在各分项的append之前，则输出的是离当期最近的fc个点的预测值；y最后一个点不能用，因为是距当期fc+1个点的预测值。
                # a、b序列的最后一个值不能用，即取到序列的倒数第二个值为止；而Y.append在a,b,s,y.append之前，所以a,b,s的索引为-1,-1,-m。
                Y.append((a[-1] + b[-1]) * s[-m])
            # s序列取到的元素需要恰好比a,Y序列的元素在索引上向前推一个周期。
            a.append(para.x[0] * (Y[i] / s[-m]) + (1 - para.x[0]) * (a[i] + b[i]))
            b.append(para.x[1] * (a[i + 1] - a[i]) + (1 - para.x[1]) * b[i])
            # 右边s序列取到的元素需要恰好比a,Y,s.append后的序列的元素在索引上向前推一个周期。
            s.append(para.x[2] * (Y[i] / a[i + 1]) + (1 - para.x[2]) * s[-m])
            # a[i+1]，b[i+1]代表Y[i]的信息，用来计算y[i+1]；又因为y含有人为赋值但不可用初值y[0]，所以s[-m]正好比y[i+1]向前推一个周期。
            y.append((a[i + 1] + b[i + 1]) * s[-m])
        if not ((len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s) == m + len(x) + fc)):
            print(s)
            raise Exception('分项长度有误，不能使用季节项序列')
    elif model == 'double_seasonal_add_add':
        for i in range(len(Y) + fc):
            if i == len(Y):
                Y.append(a[-1] + b[-1] + s_year[-m_year] + s_week[-m_week])
            a.append(para.x[0] * (Y[i] - (s_year[-m_year] + s_week[-m_week])) + (1 - para.x[0]) * (a[i] + b[i]))
            b.append(para.x[1] * (a[i + 1] - a[i]) + (1 - para.x[1]) * b[i])
            s_year.append(para.x[2] * (Y[i] - (a[i + 1] + s_week[-m_week])) + (1 - para.x[2]) * s_year[-m_year])
            s_week.append(para.x[3] * (Y[i] - (a[i + 1] + s_year[-m_year])) + (1 - para.x[3]) * s_week[-m_week])
            y.append(a[i + 1] + b[i + 1] + s_year[-m_year] + s_week[-m_week])
        if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (
                len(s_week) == m_week + len(x) + fc):
            s_total = (np.array(s_year[m_year - m_week:]) + np.array(s_week[:])).tolist()
        else:
            print(s_total)
            raise Exception('分项长度有误，不能使用季节项序列')
    elif model == 'double_seasonal_add_mul':
        for i in range(len(Y) + fc):
            if i == len(Y):
                Y.append((a[-1] + b[-1] + s_year[-m_year]) * s_week[-m_week])
            a.append(para.x[0] * (Y[i] / s_week[-m_week]  - s_year[-m_year]) + (1 - para.x[0]) * (a[i] + b[i]))
            b.append(para.x[1] * (a[i + 1] - a[i]) + (1 - para.x[1]) * b[i])
            s_year.append(para.x[2] * (Y[i] / s_week[-m_week] - a[i + 1]) + (1 - para.x[2]) * s_year[-m_year])
            s_week.append(para.x[3] * (Y[i] / (s_year[-m_year] + a[i + 1])) + (1 - para.x[3]) * s_week[-m_week])
            y.append((a[i + 1] + b[i + 1] + s_year[-m_year]) * s_week[-m_week])
        if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (
                len(s_week) == m_week + len(x) + fc):
            s_total = (np.array(s_year[m_year - m_week:]) + np.array(s_week[:])).tolist()
        else:
            print(s_total)
            raise Exception('分项长度有误，不能使用季节项序列')
    elif model == 'double_seasonal_mul_add':
        for i in range(len(Y) + fc):
            if i == len(Y):
                Y.append((a[-1] + b[-1]) * s_year[-m_year] + s_week[-m_week])
            a.append(para.x[0] * ((Y[i] - s_week[-m_week]) / s_year[-m_year]) + (1 - para.x[0]) * (a[i] + b[i]))
            b.append(para.x[1] * (a[i + 1] - a[i]) + (1 - para.x[1]) * b[i])
            s_year.append(para.x[2] * ((Y[i] - s_week[-m_week]) / a[i + 1]) + (1 - para.x[2]) * s_year[-m_year])
            s_week.append(para.x[3] * (Y[i] - s_year[-m_year] * a[i + 1]) + (1 - para.x[3]) * s_week[-m_week])
            y.append((a[i + 1] + b[i + 1]) * s_year[-m_year] + s_week[-m_week])
        if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (
                len(s_week) == m_week + len(x) + fc):
            s_total = (np.array(s_year[m_year - m_week:]) + np.array(s_week[:])).tolist()
        else:
            print(s_total)
            raise Exception('分项长度有误，不能使用季节项序列')
    elif model == 'double_seasonal_mul_mul':
        for i in range(len(Y) + fc):
            if i == len(Y):
                Y.append((a[-1] + b[-1]) * s_year[-m_year] * s_week[-m_week])
            a.append(para.x[0] * (Y[i] / (s_week[-m_week] * s_year[-m_year])) + (1 - para.x[0]) * (a[i] + b[i]))
            b.append(para.x[1] * (a[i + 1] - a[i]) + (1 - para.x[1]) * b[i])
            s_year.append(para.x[2] * (Y[i] / (s_week[-m_week] * a[i + 1])) + (1 - para.x[2]) * s_year[-m_year])
            s_week.append(para.x[3] * (Y[i] / (s_year[-m_year] * a[i + 1])) + (1 - para.x[3]) * s_week[-m_week])
            y.append((a[i + 1] + b[i + 1]) * s_year[-m_year] * s_week[-m_week])
        if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (
                len(s_week) == m_week + len(x) + fc):
            s_total = (np.array(s_year[m_year - m_week:]) + np.array(s_week[:])).tolist()
        else:
            print(s_total)
            raise Exception('分项长度有误，不能使用季节项序列')
    elif model == 'multi_seasonal_AAAA':
        for i in range(len(Y) + fc):
            if i == len(Y):
                Y.append(a[-1] + b[-1] + s_year[-m_year] + s_week[-m_week] + s_month[-m_month] + s_quarter[-m_quarter])
            a.append(para.x[0] * (Y[i] - (s_year[-m_year] + s_week[-m_week] + s_month[-m_month] + s_quarter[-m_quarter]))
                     + (1 - para.x[0]) * (a[i] + b[i]))
            b.append(para.x[1] * (a[i + 1] - a[i]) + (1 - para.x[1]) * b[i])
            s_year.append(para.x[2] * (Y[i] - (a[i + 1] + s_week[-m_week] + s_month[-m_month] + s_quarter[-m_quarter]))
                          + (1 - para.x[2]) * s_year[-m_year])
            s_week.append(para.x[3] * (Y[i] - (a[i + 1] + s_year[-m_year] + s_month[-m_month] + s_quarter[-m_quarter]))
                          + (1 - para.x[3]) * s_week[-m_week])
            s_month.append(para.x[4] * (Y[i] - (a[i + 1] + s_week[-m_week] + s_year[-m_year] + s_quarter[-m_quarter]))
                          + (1 - para.x[4]) * s_month[-m_month])
            s_quarter.append(para.x[5] * (Y[i] - (a[i + 1] + s_year[-m_year] + s_month[-m_month] + s_week[-m_week]))
                          + (1 - para.x[5]) * s_quarter[-m_quarter])
            y.append(a[i + 1] + b[i + 1] + s_year[-m_year] + s_week[-m_week] + s_month[-m_month] + s_quarter[-m_quarter])
        if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (
                len(s_quarter) == m_quarter + len(x) + fc) and (len(s_month) == m_month + len(x) + fc) and (
                len(s_week) == m_week + len(x) + fc):
            s_total = (np.array(s_year[m_year - m_week:]) + np.array(s_quarter[m_quarter - m_week:]) + np.array(
                s_month[m_month - m_week:]) + np.array(s_week[:])).tolist()
        else:
            print(s_total)
            raise Exception('分项长度有误，不能使用季节项序列')
    elif model == 'multi_seasonal_AMAA':
        for i in range(len(Y) + fc):
            if i == len(Y):
                Y.append((a[-1] + b[-1]) * s_week[-m_week] + s_year[-m_year] + s_month[-m_month] + s_quarter[-m_quarter])
            a.append(para.x[0] * ((Y[i] - (s_year[-m_year] + s_month[-m_month] + s_quarter[-m_quarter])) / s_week[-m_week])
                     + (1 - para.x[0]) * (a[i] + b[i]))
            b.append(para.x[1] * (a[i + 1] - a[i]) + (1 - para.x[1]) * b[i])
            s_year.append(para.x[2] * (Y[i] - (a[i + 1] * s_week[-m_week] + s_month[-m_month] + s_quarter[-m_quarter]))
                          + (1 - para.x[2]) * s_year[-m_year])
            s_week.append(para.x[3] * ((Y[i] - (s_year[-m_year] + s_month[-m_month] + s_quarter[-m_quarter])) / a[i + 1])
                          + (1 - para.x[3]) * s_week[-m_week])
            s_month.append(para.x[4] * (Y[i] - (a[i + 1] * s_week[-m_week] + s_year[-m_year] + s_quarter[-m_quarter]))
                          + (1 - para.x[4]) * s_month[-m_month])
            s_quarter.append(para.x[5] * (Y[i] - (a[i + 1] * s_week[-m_week] + s_year[-m_year] + s_month[-m_month]))
                          + (1 - para.x[5]) * s_quarter[-m_quarter])
            y.append((a[i + 1] + b[i + 1]) * s_week[-m_week] + s_year[-m_year] + s_month[-m_month] + s_quarter[-m_quarter])
        if (len(a) == len(b) == len(y) == 1 + len(x) + fc) and (len(s_year) == m_year + len(x) + fc) and (
                len(s_quarter) == m_quarter + len(x) + fc) and (len(s_month) == m_month + len(x) + fc) and (
                len(s_week) == m_week + len(x) + fc):
            s_total = (np.array(s_year[m_year - m_week:]) + np.array(s_quarter[m_quarter - m_week:]) + np.array(
                s_month[m_month - m_week:]) + np.array(s_week[:])).tolist()
        else:
            print(s_total)
            raise Exception('分项长度有误，不能使用季节项序列')



    elif model == 'multi_seasonal_MAAA':
        pass
    elif model == 'multi_seasonal_MMAA':
        pass

    if sum(np.array(Y[-fc:]) - np.array(y[-fc-1:-1])) != 0:
        print(Y[-fc:], y[-fc-1:-1])
        raise Exception('预测值Y和y不相等')

    '''拟合程度指标，只需比较历史期，不能比较预测期上的指标，因为历史期上才有真实销量，预测期上无真实销量'''
    rmse_index = rmse(Y[1:-fc], y[1:-fc - 1])
    aic_index = aic(Y[1:-fc], y[1:-fc - 1], len(para.x))
    std_residual = np.nanstd([(m - n) for m, n in zip(Y[1:-fc], y[1:-fc - 1])])

    return {'predict': y[-fc-1:-1], 'alpha': alpha, 'beta': beta, 'gamma_year': gamma_year, 'gamma_week': gamma_week,
            'gamma_month': gamma_month, 'gamma_quarter': gamma_month,
            'rmse': rmse_index, 'aic': aic_index, 'std_of_residual': std_residual, 'res': rmse_index - std_residual,
            'fittedvalues': y[:-fc - 1], 'weights': weights,

            's_total': s_total, 'len(s_total)': len(s_total), 'max(s_total)': max(s_total), 'min(s_total)': min(s_total),
            'mean(s_total)': np.mean(s_total), 'np.std(s_total)': np.std(s_total), 's_total_75': np.percentile(s_total, 75),
            's_total_25': np.percentile(s_total, 25), 's_total_50': np.percentile(s_total, 50),

            's_year': s_year, 'len(s_year)': len(s_year), 'max(s_year)': max(s_year), 'min(s_year)': min(s_year),
            'mean(s_year)': np.mean(s_year), 'np.std(s_year)': np.std(s_year), 's_year_75': np.percentile(s_year, 75),
            's_year_25': np.percentile(s_year, 25), 's_year_50': np.percentile(s_year, 50),

            's_week': s_week, 'len(s_week)': len(s_week), 'max(s_week)': max(s_week), 'min(s_week)': min(s_week),
            'mean(s_week)': np.mean(s_week), 'np.std(s_week)': np.std(s_week), 's_week_75': np.percentile(s_week, 75),
            's_week_25': np.percentile(s_week, 25), 's_week_50': np.percentile(s_week, 50),

            's_month': s_month, 'len(s_month)': len(s_month), 'max(s_month)': max(s_month), 'min(s_month)': min(s_month),
            'mean(s_month)': np.mean(s_month), 'np.std(s_month)': np.std(s_month), 's_year_75': np.percentile(s_month, 75),
            's_year_25': np.percentile(s_month, 25), 's_year_50': np.percentile(s_month, 50),

            's_quarter': s_quarter, 'len(s_quarter)': len(s_quarter), 'max(s_quarter)': max(s_quarter), 'min(s_quarter)': min(s_quarter),
            'mean(s_quarter)': np.mean(s_quarter), 'np.std(s_quarter)': np.std(s_quarter), 's_week_75': np.percentile(s_quarter, 75),
            's_week_25': np.percentile(s_quarter, 25), 's_week_50': np.percentile(s_quarter, 50)}

ExponentialSmoothingModels([1,2,3], granularity='day',)
# ExponentialSmoothingModels([1,2,3], granularity='day', model='simple', initial_values=[0.2], boundaries=[(0,1)])

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
    g1 = multi_seasonal_add_day(day)
    print('multiseasonal_add_day', g1, '\n')
    g2 = multi_seasonal_add_week(week)
    print('multiseasonal_add_week', g2, '\n')
    g3 = multi_seasonal_mul_day(day)
    print('multiseasonal_mul_day', g3, '\n')
    g4 = multi_seasonal_mul_week(week)
    print('multiseasonal_mul_week', g4)
