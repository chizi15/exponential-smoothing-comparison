import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
import math
from scipy import stats
from time import process_time
from pandas import DataFrame
from scipy import optimize
import numpy as np
import random
from warnings import filterwarnings
filterwarnings("ignore")


###########-------setting up input data-----------------######################
up_limit = 20 # 设置level、trend、season项的取值上限
steps_day, steps_week = 28, 4
length = [730+steps_day, 365+steps_day, 104+steps_week, 52+steps_week] # 代表每个序列的长度，分别为周、日序列的一年及两年。
y_level, y_trend, y_season, y_noise, y_input = [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length)

weights = []
for i in range(-up_limit+1, 1):
    weights.append(0.6 ** i)  # 设置y_level项随机序列的权重呈递减指数分布，底数越小，y_level中较小值所占比例越大。
weights = np.array(weights)

y_season[0] = up_limit/4 * (1 + np.sin(np.linspace(0, 2*2*np.pi, length[0]))) # 用正弦函数模拟季节性
y_season[1] = up_limit/4 * (1 + np.sin(np.linspace(0, 2*np.pi, length[1])))
y_season[2] = up_limit/4 * (1 + np.sin(np.linspace(0, 2*2*np.pi, length[2])))
y_season[3] = up_limit/4 * (1 + np.sin(np.linspace(0, 2*np.pi, length[3])))
for i in range(0, len(length)):
    y_level[i] = np.array(random.choices(range(0, up_limit), weights=weights, k=length[i]))  # 用指数权重分布随机数模拟基础项
    y_trend[i] = np.log2(np.linspace(2, 2**(up_limit/4), num=length[i])) + (min(np.log2(np.linspace(2, 2**(up_limit/4), num=length[i]))) +
                 max(np.log2(np.linspace(2, 2**(up_limit/4), num=length[i])))) / length[i] * np.linspace(1, length[i], num=length[i]) # 用对数函数与线性函数的均值模拟趋势性
    y_noise[i] = np.random.normal(0, 1, length[i]) # 假定数据处于理想状态，用正态分布模拟噪音
    y_input[i] = y_level[i] + y_trend[i] + y_season[i] + y_noise[i] # 假定各项以加法方式组成输入数据

    y_level[i] = pd.Series(y_level[i]).rename('y_level')
    y_trend[i] = pd.Series(y_trend[i]).rename('y_trend')
    y_season[i] = pd.Series(y_season[i]).rename('y_season')
    y_noise[i] = pd.Series(y_noise[i]).rename('y_noise')
    y_input[i] = pd.Series(y_input[i]).rename('y_input')
    y_input[i][y_input[i] < 0] = 0
###########-------setting up input data-----------------######################


###########-------plot input data-----------------######################
plt.figure('730+28', figsize=(20,10))
ax1 = plt.subplot(5,1,1)
ax2 = plt.subplot(5,1,2)
ax3 = plt.subplot(5,1,3)
ax4 = plt.subplot(5,1,4)
ax5 = plt.subplot(5,1,5)
y_input[0].plot(ax=ax1, legend=True)
y_level[0].plot(ax=ax2, legend=True)
y_trend[0].plot(ax=ax3, legend=True)
y_season[0].plot(ax=ax4, legend=True)
y_noise[0].plot(ax=ax5, legend=True)

plt.figure('365+28', figsize=(20,10))
ax1 = plt.subplot(5,1,1)
ax2 = plt.subplot(5,1,2)
ax3 = plt.subplot(5,1,3)
ax4 = plt.subplot(5,1,4)
ax5 = plt.subplot(5,1,5)
y_input[1].plot(ax=ax1, legend=True)
y_level[1].plot(ax=ax2, legend=True)
y_trend[1].plot(ax=ax3, legend=True)
y_season[1].plot(ax=ax4, legend=True)
y_noise[1].plot(ax=ax5, legend=True)

plt.figure('104+4', figsize=(20,10))
ax1 = plt.subplot(5,1,1)
ax2 = plt.subplot(5,1,2)
ax3 = plt.subplot(5,1,3)
ax4 = plt.subplot(5,1,4)
ax5 = plt.subplot(5,1,5)
y_input[2].plot(ax=ax1, legend=True)
y_level[2].plot(ax=ax2, legend=True)
y_trend[2].plot(ax=ax3, legend=True)
y_season[2].plot(ax=ax4, legend=True)
y_noise[2].plot(ax=ax5, legend=True)

plt.figure('52+4', figsize=(20,10))
ax1 = plt.subplot(5,1,1)
ax2 = plt.subplot(5,1,2)
ax3 = plt.subplot(5,1,3)
ax4 = plt.subplot(5,1,4)
ax5 = plt.subplot(5,1,5)
y_input[3].plot(ax=ax1, legend=True)
y_level[3].plot(ax=ax2, legend=True)
y_trend[3].plot(ax=ax3, legend=True)
y_season[3].plot(ax=ax4, legend=True)
y_noise[3].plot(ax=ax5, legend=True)
###########-------plot input data-----------------######################


###################------SES------------#########################
fit_SES = SimpleExpSmoothing(y_input[0][0:730]).fit(smoothing_level=0.2)
fit_SES_train = SimpleExpSmoothing(y_input[0][0:730]).fit(optimized=True, use_brute=False) # start_params取上一轮训练得到的alpha更好

print()
print('SSE_730:')
results=pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SSE"])
params = ['smoothing_level', 'initial_level']
results["fit_SES"] = [fit_SES.params[p] for p in params] + [fit_SES.sse]
results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
print(results)

plt.figure('730+28+SES', figsize=(20,10))
ax_SES = y_input[0].rename('y_input[0]').plot(color='k', legend=True)
ax_SES.set_ylabel("amount")
ax_SES.set_xlabel("day")
fit_SES.fittedvalues.plot(ax=ax_SES, color='b')
fit_SES_train.fittedvalues.plot(ax=ax_SES, color='r')
fit_SES.forecast(steps_day).rename(r'$\alpha=%s$'%fit_SES.model.params['smoothing_level']).plot(ax=ax_SES, color='b', legend=True)
fit_SES_train.forecast(steps_day).rename(r'$\alpha=%s$'%fit_SES_train.model.params['smoothing_level']).plot(ax=ax_SES, color='r', legend=True)

##############################

fit_SES = SimpleExpSmoothing(y_input[1][0:365]).fit(smoothing_level=0.2)
fit_SES_train = SimpleExpSmoothing(y_input[1][0:365]).fit(optimized=True, use_brute=False)

print()
print('SSE_365:')
results=pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SSE"])
params = ['smoothing_level', 'initial_level']
results["fit_SES"] = [fit_SES.params[p] for p in params] + [fit_SES.sse]
results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
print(results)

plt.figure('365+28+SES', figsize=(20,10))
ax_SES = y_input[1].rename('y_input[1]').plot(color='black', legend=True)
ax_SES.set_ylabel("amount")
ax_SES.set_xlabel("day")
fit_SES.fittedvalues.plot(ax=ax_SES, color='blue')
fit_SES_train.fittedvalues.plot(ax=ax_SES, color='red')
fit_SES.forecast(steps_day).rename(r'$\alpha=%s$'%fit_SES.model.params['smoothing_level']).plot(ax=ax_SES, color='b', legend=True)
fit_SES_train.forecast(steps_day).rename(r'$\alpha=%s$'%fit_SES_train.model.params['smoothing_level']).plot(ax=ax_SES, color='red', legend=True)

##############################

fit_SES = SimpleExpSmoothing(y_input[2][0:104]).fit(smoothing_level=0.2)
fit_SES_train = SimpleExpSmoothing(y_input[2][0:104]).fit(optimized=True, use_brute=False)

print()
print('SSE_104:')
results=pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SSE"])
params = ['smoothing_level', 'initial_level']
results["fit_SES"] = [fit_SES.params[p] for p in params] + [fit_SES.sse]
results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
print(results)

plt.figure('104+4+SES', figsize=(20,10))
ax_SES = y_input[2].rename('y_input[2]').plot(color='black', legend=True)
ax_SES.set_ylabel("amount")
ax_SES.set_xlabel("week")
fit_SES.fittedvalues.plot(ax=ax_SES, color='blue')
fit_SES_train.fittedvalues.plot(ax=ax_SES, color='red')
fit_SES.forecast(steps_week).rename(r'$\alpha=%s$'%fit_SES.model.params['smoothing_level']).plot(ax=ax_SES, color='b', legend=True)
fit_SES_train.forecast(steps_week).rename(r'$\alpha=%s$'%fit_SES_train.model.params['smoothing_level']).plot(ax=ax_SES, color='red', legend=True)

##############################

fit_SES = SimpleExpSmoothing(y_input[3][0:52]).fit(smoothing_level=0.2)
fit_SES_train = SimpleExpSmoothing(y_input[3][0:52]).fit(optimized=True, use_brute=False)

print()
print('SSE_52:')
results=pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SSE"])
params = ['smoothing_level', 'initial_level']
results["fit_SES"] = [fit_SES.params[p] for p in params] + [fit_SES.sse]
results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
print(results)

plt.figure('52+4+SES', figsize=(20,10))
ax_SES = y_input[3].rename('y_input[3]').plot(color='black', legend=True)
ax_SES.set_ylabel("amount")
ax_SES.set_xlabel("week")
fit_SES.fittedvalues.plot(ax=ax_SES, color='blue')
fit_SES_train.fittedvalues.plot(ax=ax_SES, color='red')
fit_SES.forecast(steps_week).rename(r'$\alpha=%s$'%fit_SES.model.params['smoothing_level']).plot(ax=ax_SES, color='b', legend=True)
fit_SES_train.forecast(steps_week).rename(r'$\alpha=%s$'%fit_SES_train.model.params['smoothing_level']).plot(ax=ax_SES, color='red', legend=True)
###################------SES------------#########################


###################------Holt------------#########################
Holt_add_dam = Holt(y_input[0][0:730], exponential=False, damped=True).fit(smoothing_level=0.1, smoothing_slope=0.2, damping_slope=0.92, optimized=False)
Holt_add_dam_train = Holt(y_input[0][0:730], exponential=False, damped=True).fit(optimized=True, use_brute=False,)
Holt_mul_dam = Holt(y_input[0][0:730], exponential=True, damped=True).fit(smoothing_level=0.1, smoothing_slope=0.2, damping_slope=0.92, optimized=False)
Holt_mul_dam_train = Holt(y_input[0][0:730], exponential=True, damped=True).fit(optimized=True, use_brute=False) # start_params取上一轮训练得到的alpha/beta/phi更好

print()
print('Hlot_730:')
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"])
params = ['smoothing_level', 'smoothing_slope', 'damping_slope', 'initial_level', 'initial_slope']
results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
results["Holt_mul_dam"] = [Holt_mul_dam.params[p] for p in params] + [Holt_mul_dam.sse]
results["Holt_mul_dam_train"] = [Holt_mul_dam_train.params[p] for p in params] + [Holt_mul_dam_train.sse]
print(results)

plt.figure('730+28+Holt_dam', figsize=(20,10))
ax_Holt = y_input[0].rename('y_input[0]').plot(color='black', legend=True)
ax_Holt.set_ylabel("amount")
ax_Holt.set_xlabel("day")
Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
Holt_mul_dam.fittedvalues.plot(ax=ax_Holt, color='g')
Holt_mul_dam_train.fittedvalues.plot(ax=ax_Holt, color='y')
Holt_add_dam.forecast(steps_day).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
Holt_add_dam_train.forecast(steps_day).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)
Holt_mul_dam.forecast(steps_day).rename('Holt_mul_dam').plot(ax=ax_Holt, color='g', legend=True)
Holt_mul_dam_train.forecast(steps_day).rename('Holt_mul_dam_train').plot(ax=ax_Holt, color='y', legend=True)

##############################

Holt_add_dam = Holt(y_input[1][0:365], exponential=False, damped=True).fit(smoothing_level=0.15, smoothing_slope=0.25, damping_slope=0.94, optimized=False)
Holt_add_dam_train = Holt(y_input[1][0:365], exponential=False, damped=True).fit(optimized=True, use_brute=False,)
Holt_mul_dam = Holt(y_input[1][0:365], exponential=True, damped=True).fit(smoothing_level=0.15, smoothing_slope=0.25, damping_slope=0.94, optimized=False)
Holt_mul_dam_train = Holt(y_input[1][0:365], exponential=True, damped=True).fit(optimized=True, use_brute=False) # start_params取上一轮训练得到的alpha/beta/phi更好

print()
print('Hlot_365:')
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"])
params = ['smoothing_level', 'smoothing_slope', 'damping_slope', 'initial_level', 'initial_slope']
results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
results["Holt_mul_dam"] = [Holt_mul_dam.params[p] for p in params] + [Holt_mul_dam.sse]
results["Holt_mul_dam_train"] = [Holt_mul_dam_train.params[p] for p in params] + [Holt_mul_dam_train.sse]
print(results)

plt.figure('365+28+Holt_dam', figsize=(20,10))
ax_Holt = y_input[1].rename('y_input[1]').plot(color='black', legend=True)
ax_Holt.set_ylabel("amount")
ax_Holt.set_xlabel("day")
Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
Holt_mul_dam.fittedvalues.plot(ax=ax_Holt, color='g')
Holt_mul_dam_train.fittedvalues.plot(ax=ax_Holt, color='y')
Holt_add_dam.forecast(steps_day).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
Holt_add_dam_train.forecast(steps_day).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)
Holt_mul_dam.forecast(steps_day).rename('Holt_mul_dam').plot(ax=ax_Holt, color='g', legend=True)
Holt_mul_dam_train.forecast(steps_day).rename('Holt_mul_dam_train').plot(ax=ax_Holt, color='y', legend=True)

##############################

Holt_add_dam = Holt(y_input[2][0:104], exponential=False, damped=True).fit(smoothing_level=0.2, smoothing_slope=0.3, damping_slope=0.96, optimized=False)
Holt_add_dam_train = Holt(y_input[2][0:104], exponential=False, damped=True).fit(optimized=True, use_brute=False,)
Holt_mul_dam = Holt(y_input[2][0:104], exponential=True, damped=True).fit(smoothing_level=0.2, smoothing_slope=0.3, damping_slope=0.96, optimized=False)
Holt_mul_dam_train = Holt(y_input[2][0:104], exponential=True, damped=True).fit(optimized=True, use_brute=False) # start_params取上一轮训练得到的alpha/beta/phi更好

print()
print('Hlot_104:')
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"])
params = ['smoothing_level', 'smoothing_slope', 'damping_slope', 'initial_level', 'initial_slope']
results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
results["Holt_mul_dam"] = [Holt_mul_dam.params[p] for p in params] + [Holt_mul_dam.sse]
results["Holt_mul_dam_train"] = [Holt_mul_dam_train.params[p] for p in params] + [Holt_mul_dam_train.sse]
print(results)

plt.figure('104+4+Holt_dam', figsize=(20,10))
ax_Holt = y_input[2].rename('y_input[2]').plot(color='black', legend=True)
ax_Holt.set_ylabel("amount")
ax_Holt.set_xlabel("week")
Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
Holt_mul_dam.fittedvalues.plot(ax=ax_Holt, color='g')
Holt_mul_dam_train.fittedvalues.plot(ax=ax_Holt, color='y')
Holt_add_dam.forecast(steps_week).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
Holt_add_dam_train.forecast(steps_week).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)
Holt_mul_dam.forecast(steps_week).rename('Holt_mul_dam').plot(ax=ax_Holt, color='g', legend=True)
Holt_mul_dam_train.forecast(steps_week).rename('Holt_mul_dam_train').plot(ax=ax_Holt, color='y', legend=True)

##############################

Holt_add_dam = Holt(y_input[3][0:52], exponential=False, damped=True).fit(smoothing_level=0.25, smoothing_slope=0.35, damping_slope=0.98, optimized=False)
Holt_add_dam_train = Holt(y_input[3][0:52], exponential=False, damped=True).fit(optimized=True, use_brute=False,)
Holt_mul_dam = Holt(y_input[3][0:52], exponential=True, damped=True).fit(smoothing_level=0.25, smoothing_slope=0.35, damping_slope=0.98, optimized=False)
Holt_mul_dam_train = Holt(y_input[3][0:52], exponential=True, damped=True).fit(optimized=True, use_brute=False) # start_params取上一轮训练得到的alpha/beta/phi更好

print()
print('Hlot_52:')
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"])
params = ['smoothing_level', 'smoothing_slope', 'damping_slope', 'initial_level', 'initial_slope']
results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
results["Holt_mul_dam"] = [Holt_mul_dam.params[p] for p in params] + [Holt_mul_dam.sse]
results["Holt_mul_dam_train"] = [Holt_mul_dam_train.params[p] for p in params] + [Holt_mul_dam_train.sse]
print(results)

plt.figure('52+4+Holt_dam', figsize=(20,10))
ax_Holt = y_input[3].rename('y_input[3]').plot(color='black', legend=True)
ax_Holt.set_ylabel("amount")
ax_Holt.set_xlabel("week")
Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
Holt_mul_dam.fittedvalues.plot(ax=ax_Holt, color='g')
Holt_mul_dam_train.fittedvalues.plot(ax=ax_Holt, color='y')
Holt_add_dam.forecast(steps_week).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
Holt_add_dam_train.forecast(steps_week).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)
Holt_mul_dam.forecast(steps_week).rename('Holt_mul_dam').plot(ax=ax_Holt, color='g', legend=True)
Holt_mul_dam_train.forecast(steps_week).rename('Holt_mul_dam_train').plot(ax=ax_Holt, color='y', legend=True)
###################------Holt------------#########################


###################------HoltWinters------------#########################
HW_add_add_dam = ExponentialSmoothing(y_input[0][0:730], seasonal_periods=365, trend='add', seasonal='add', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)
HW_add_mul_dam = ExponentialSmoothing(y_input[0][0:730], seasonal_periods=365, trend='add', seasonal='mul', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)
HW_mul_add_dam = ExponentialSmoothing(y_input[0][0:730], seasonal_periods=365, trend='mul', seasonal='add', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)
HW_mul_mul_dam = ExponentialSmoothing(y_input[0][0:730], seasonal_periods=365, trend='mul', seasonal='mul', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)

print()
print('HoltWinters_730:')
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$\gamma$",r"$l_0$","$b_0$","SSE"])
params = ['smoothing_level', 'smoothing_slope', 'damping_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
print(results)

plt.figure('730+28+HoltWinters', figsize=(20,10))
ax_HoltWinters = y_input[0].rename('y_input[0]').plot(color='k', legend='True')
ax_HoltWinters.set_ylabel("amount")
ax_HoltWinters.set_xlabel("day")
HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
HW_add_add_dam.forecast(steps_day).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
HW_add_mul_dam.forecast(steps_day).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
HW_mul_add_dam.forecast(steps_day).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
HW_mul_mul_dam.forecast(steps_day).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)

##############################

HW_add_add_dam = ExponentialSmoothing(y_input[1][0:365+1], seasonal_periods=365, trend='add', seasonal='add', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)
HW_add_mul_dam = ExponentialSmoothing(y_input[1][0:365+1], seasonal_periods=365, trend='add', seasonal='mul', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)
HW_mul_add_dam = ExponentialSmoothing(y_input[1][0:365+1], seasonal_periods=365, trend='mul', seasonal='add', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)
HW_mul_mul_dam = ExponentialSmoothing(y_input[1][0:365+1], seasonal_periods=365, trend='mul', seasonal='mul', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)

print()
print('HoltWinters_365:')
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$\gamma$",r"$l_0$","$b_0$","SSE"])
params = ['smoothing_level', 'smoothing_slope', 'damping_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
print(results)

plt.figure('365+28+HoltWinters', figsize=(20,10))
ax_HoltWinters = y_input[1].rename('y_input[1]').plot(color='k', legend='True')
ax_HoltWinters.set_ylabel("amount")
ax_HoltWinters.set_xlabel("day")
HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
HW_add_add_dam.forecast(steps_day-1).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
HW_add_mul_dam.forecast(steps_day-1).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
HW_mul_add_dam.forecast(steps_day-1).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
HW_mul_mul_dam.forecast(steps_day-1).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)

##############################

HW_add_add_dam = ExponentialSmoothing(y_input[2][0:104], seasonal_periods=52, trend='add', seasonal='add', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)
HW_add_mul_dam = ExponentialSmoothing(y_input[2][0:104], seasonal_periods=52, trend='add', seasonal='mul', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)
HW_mul_add_dam = ExponentialSmoothing(y_input[2][0:104], seasonal_periods=52, trend='mul', seasonal='add', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)
HW_mul_mul_dam = ExponentialSmoothing(y_input[2][0:104], seasonal_periods=52, trend='mul', seasonal='mul', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)

print()
print('HoltWinters_104:')
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$\gamma$",r"$l_0$","$b_0$","SSE"])
params = ['smoothing_level', 'smoothing_slope', 'damping_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
print(results)

plt.figure('104+4+HoltWinters', figsize=(20,10))
ax_HoltWinters = y_input[2].rename('y_input[2]').plot(color='k', legend='True')
ax_HoltWinters.set_ylabel("amount")
ax_HoltWinters.set_xlabel("day")
HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
HW_add_add_dam.forecast(steps_week).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
HW_add_mul_dam.forecast(steps_week).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
HW_mul_add_dam.forecast(steps_week).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
HW_mul_mul_dam.forecast(steps_week).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)

##############################

HW_add_add_dam = ExponentialSmoothing(y_input[3][0:52+1], seasonal_periods=52, trend='add', seasonal='add', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)
HW_add_mul_dam = ExponentialSmoothing(y_input[3][0:52+1], seasonal_periods=52, trend='add', seasonal='mul', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)
HW_mul_add_dam = ExponentialSmoothing(y_input[3][0:52+1], seasonal_periods=52, trend='mul', seasonal='add', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)
HW_mul_mul_dam = ExponentialSmoothing(y_input[3][0:52+1], seasonal_periods=52, trend='mul', seasonal='mul', damped=True).\
    fit(use_boxcox=True, use_basinhopping=False, use_brute=False)

print()
print('HoltWinters_52:')
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$\gamma$",r"$l_0$","$b_0$","SSE"])
params = ['smoothing_level', 'smoothing_slope', 'damping_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
print(results)

plt.figure('52+4+HoltWinters', figsize=(20,10))
ax_HoltWinters = y_input[3].rename('y_input[3]').plot(color='k', legend='True')
ax_HoltWinters.set_ylabel("amount")
ax_HoltWinters.set_xlabel("day")
HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
HW_add_add_dam.forecast(steps_week-1).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
HW_add_mul_dam.forecast(steps_week-1).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
HW_mul_add_dam.forecast(steps_week-1).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
HW_mul_mul_dam.forecast(steps_week-1).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)
###################------HoltWinters------------#########################
