#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
import pyecharts.charts as echarts
from pyecharts.charts import Bar
from pyecharts import options as opts


# In[143]:


# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sys, os, time, copy
import gensim
from tqdm import tqdm_notebook as tqdm

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
import gc
import os, random, re, copy, pickle
from urllib.parse import urlparse

from scipy.stats import spearmanr
from math import floor, ceil
import keras.backend.tensorflow_backend as ktf

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Lambda, LSTM, TimeDistributed, Masking, Bidirectional, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Reshape, Flatten, Dropout, concatenate, Concatenate, Add, add, RepeatVector, Permute, multiply,Activation, BatchNormalization
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import multi_gpu_model,to_categorical, Sequence
from tensorflow.keras.initializers import RandomUniform, RandomNormal, glorot_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model

from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
np.set_printoptions(suppress=True)


# In[144]:


#%% 
## set global parameters
random_seed = 2020
random.seed(random_seed)
np.random.seed(random_seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
gpus = tf.config.experimental.list_physical_devices('GPU')
ngpus = 1
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)


# In[117]:


elder_add_rate = [0.04]
for i in range(len(elder_num)-1):
    elder_add_rate.append((elder_num[i+1]-elder_num[i])/elder_num[i])
elder_add_rate = (np.array(elder_add_rate)*100).tolist()

bed_add_rate = [0.071]
for i in range(len(bed_num)-1):
    bed_add_rate.append((bed_num[i+1]-bed_num[i])/bed_num[i])
bed_add_rate = (np.array(bed_add_rate)*100).tolist()


# In[121]:


import pyecharts.options as opts
from pyecharts.charts import Bar, Line

bed_num = np.array([
            3.496,
            3.964,
            4.493,
            5.267,
            4.823,
            3.932,
            4.140,
            4.196,
        ])
elder_num = np.array([
            177.65,
            184.99,
            193.90,
            202.43,
            212.42,
            222.00,
            230.86,
            240.90,
        ])
x_data = ["2010年", "2011年", "2012年", "2013年", "2014年", "2015年", "2016年", "2017年"]

bar = (
    Bar()
    .add_xaxis(xaxis_data=x_data)
    .add_yaxis(
        series_name="床位数",
        stack="stack1",
        category_gap=50,
        color = '#1c86ee',
        yaxis_data=bed_num.tolist(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="老人人口",
        stack="stack1",
        category_gap=50,
        color = 'green',
        yaxis_data=(elder_num-bed_num).tolist(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .extend_axis(
        yaxis=opts.AxisOpts(
            name="增长率",
            type_="value",
            min_=-20,
            max_=20,
#             interval=5,
            axislabel_opts=opts.LabelOpts(formatter="{value} %"),
        )
    )

    .set_global_opts(
        title_opts=opts.TitleOpts(title="老人人数及床位变化图", subtitle="2010-2017"),
        tooltip_opts=opts.TooltipOpts(
            is_show=True, trigger="axis", axis_pointer_type="cross"
        ),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="shadow"),
        ),
        yaxis_opts=opts.AxisOpts(
            name="人数",
            type_="value",
            min_=0,
            max_=250,
            interval=50,
            
            axislabel_opts=opts.LabelOpts(formatter="{value} /百万人"),
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
            
        ),
    )
)

line = (
    Line()
    .add_xaxis(xaxis_data=x_data)
    .add_yaxis(
        series_name="老人增长率",
        yaxis_index=1,
        y_axis=elder_add_rate,
        is_smooth=True,
        z_level=3,
#         symbol="triangle",
        linestyle_opts=opts.LineStyleOpts(width=4, type_="dashed"),
#         itemstyle_opts=opts.ItemStyleOpts(
#             border_width=3, border_color="yellow", color="blue"
#         ),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="床位增长率",
        yaxis_index=1,
        y_axis=bed_add_rate,
        is_smooth=True,
        z_level=3,
        linestyle_opts=opts.LineStyleOpts(width=4),
        label_opts=opts.LabelOpts(is_show=False),
    )
)

bar.overlap(line).render_notebook()
# line.overlap(bar).render_notebook()


# ## 1-2
# ### regression

# In[324]:


countryside = [14571, 13587, 13281, 12812, 12282, 11315, 10872, 10529]
town = [19410, 19683, 19881, 20117, 20401, 20515, 20883, 21116]
street = [6923, 7194, 7282, 7566, 7696, 7957, 8105, 8241]

social_services = [2697.5, 3229.1, 3683.7, 4276.5, 4404.1, 4926.4, 5440.2, 5932.7]
capital_construction = [183.0, 218.5, 235.0, 292.8, 282.2, 239.9, 243.5, 209.2]
assets = [6589.3, 6676.7, 6675.4, 6810.2, 7213.0, 8183.1, 5393.6, 5434.8]

beds = [3.496,3.964,4.493,5.267,4.823,3.932,4.140,4.196]
bed_add_rate = [7.1,13.4, 13.4, 17.2, -8.4, -18.5, 5.3, 1.4]

elder = [177.65, 184.99, 193.9 , 202.43, 212.42, 222, 230.86, 240.9 ]
elder_rate = [13.3,13.7,14.3,14.9,15.5,16.1,16.7,17.3]

adopt_num = [34529, 31424, 27278, 24460, 22772, 22348, 18736, 18820]
adopt_rate = [-22.0, -9.0, -13.2, -10.3, -6.9, -1.9, -16.2, 0.4]

city_min = [2310.5, 2276.8, 2143.5, 2064.2, 1877.0, 1701.1, 1480.2, 1261.0]
countryside_min = [5214.0, 5305.7, 5344.5, 5388.0, 5207.2, 4903.6, 4586.5, 4045.2]
special_poor = [556.3, 551.0, 545.6, 537.2, 529.1, 516.8, 496.9, 466.9]

miss = [7844, 1126, 1530, 2284, 1818, 967, 1706, 979]
donate = [596.8, 490.1, 572.5, 566.4, 604.4, 654.5, 827.0, 754.2]

sale = [968.0, 1278.0, 1510.3, 1765.3, 2059.7, 2015.1, 2064.9, 2169.8]
sale_add_rate = [28.0, 32.0, 18.2, 16.9, 16.7, -2.2, 2.5, 5.1]
Welfare_lottery_fond = [298.8, 382.0, 449.4, 510.7, 585.7, 563.8, 591.8, 631.1]

subsidy_num = [625.0, 852.5, 944.4, 950.5, 917.3, 897.0, 874.8, 857.7]
subsidy_money = [362.7, 428.3, 517.0, 618.4, 636.6, 686.8, 769.8, 827.3]
subsidy_add_rate = [16.9, 18.1, 20.7, 19.6, 2.9, 7.9, 12.1, 7.5]

Community_service = [15.3, 16.0, 20.0, 25.2, 31.1, 36.1, 38.6, 40.7]
Community_service_center = [5.7, 7.1, 10.4, 12.8, 14.3, 15.2, 16.1, 16.8]
Community_service_add_rate = [-9.8, 23.9, 47.8, 23.1, 11.7, 6.2, 5.8, 4.3]

Social_groups = [24.5, 25.5, 27.1, 28.9, 31.0, 32.9, 33.6, 35.5]
Foundation = [2200, 2614, 3029, 3549, 4117, 4784, 5559, 6307]
people_run_non_enterprise = [19.8, 20.4, 22.5, 25.5, 29.2, 32.9, 36.1, 40.0]

committee = [8.7, 8.9, 9.1, 9.5, 9.7, 10.0, 10.3, 10.6]
village_committee = [59.5, 59.0, 58.8, 58.9, 58.5, 58.1, 55.9, 55.4]

Marriage_rate = [9.3, 9.7, 9.8, 9.9, 9.6, 9.0, 8.3, 7.7]
divorce_rate = [2.0, 2.1, 2.3, 2.6, 2.7, 2.8, 3.0, 3.2]

Cremation_remains = [474.1, 468.1, 477.7, 468.9, 459.3, 459.5, 471.8, 482.0]
Cremation_rate = [49.0, 48.8, 49.5, 48.2, 47.0, 47.1, 48.3, 48.9]


# In[325]:


df_2017 = pd.DataFrame({'beds':beds, 'bed_add_rate':bed_add_rate, 'elder':elder, 'elder_rate':elder_rate,
                        'countryside':countryside, 'town':town, 'street':street, 'social_services':social_services,
                       'capital_construction':capital_construction, 'assets':assets,
                       'adopt_num':adopt_num, 'adopt_rate':adopt_rate, 'city_min':city_min, 'countryside_min':countryside_min,
                       'special_poor':special_poor, 'miss':miss, 'donate':donate, 'sale':sale, 'sale_add_rate':sale_add_rate,
                       'Welfare_lottery_fond':Welfare_lottery_fond, 'subsidy_num':subsidy_num, 'subsidy_add_rate':subsidy_add_rate,
                       'subsidy_money':subsidy_money, 'Community_service':Community_service, 'Community_service_center':Community_service_center,
                       'Community_service_add_rate':Community_service_add_rate, 'Social_groups':Social_groups,
                       'Foundation':Foundation, 'people_run_non_enterprise':people_run_non_enterprise,
                       'committee':committee, 'village_committee':village_committee,
                       'Marriage_rate':Marriage_rate, 'divorce_rate':divorce_rate,
                       'Cremation_remains':Cremation_remains, 'Cremation_rate':Cremation_rate})


# In[ ]:


[df_2017.social_services, df_2017.capital_construction, df_2017.assets]


# In[173]:


df_std = df_2017 = (df_2017-df_2017.min())/(df_2017.max()-df_2017.min())


# In[149]:


columns = ['床位数','床位数增长率','老年人口数量', 
           '老年人口比重','乡','镇','街道',
           '社会服务事业费支出','基本建设完成投资','机构和设施固定资产原价',
           '收养登记数', '收养登记数年增长率','城市低保人数','农村低保人数','农村特困人员人数',
           '因灾死亡含失踪人口','民政部门和社会组织共计接收社会捐款',
           '福利彩票销售额', '福利彩票销售额年增长率', 
           '福利彩票筹集彩票公益金','国家抚恤补助优抚对象',
           '抚恤事业费','抚恤事业费年增长率','社区服务机构和设施',
           '社区服务中心站','社区服务中心站增长率', '社会团体', 
           '基金会','民办非企业单位','居委会','村委会','结婚率','离婚率','火化遗体','火化率']
df_2017.columns = columns
df_2017


# In[168]:


df_2017.to_csv('development_statistics_2017.csv')


# In[152]:


df_std


# In[174]:


f,ax = plt.subplots(figsize=(16, 16))
sns.heatmap(df_std.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
# plt.title("社会服务发展相关系数图")
plt.show()


# In[157]:


f.savefig('社会服务发展相关系数图.png')


# In[221]:


from pyecharts.charts import Funnel,Grid

x_data = ["家庭养老", "社区养老", "机构养老"]
y_data = [60, 40, 20]

data = [[x_data[i], y_data[i]] for i in range(len(x_data))]

(
    Funnel(init_opts=opts.InitOpts(width="600px", height="500px"))
    .add(
        series_name="",
        data_pair=data,
        gap=2,
        
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a} <br/>{b} : {c}%"),
        label_opts=opts.LabelOpts(is_show=True, position="inside"),
        itemstyle_opts=opts.ItemStyleOpts(border_color="#fff", border_width=1),
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="养老结构金字塔", subtitle="", pos_left='40%', pos_top='top'),
                    legend_opts=opts.LegendOpts(pos_top='6%'),)
    .render_notebook()
)
# x_data = ["家庭", "社区", "机构"]
# y_data = [60, 30, 20]

# data = [[x_data[i], y_data[i]] for i in range(len(x_data))]

# funnel2 = (
#     Funnel(init_opts=opts.InitOpts(width="600px", height="400px"))
#     .add(
#         series_name="",
#         data_pair=data,
#         gap=2,
        
#         tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a} <br/>{b} : {c}%"),
#         label_opts=opts.LabelOpts(is_show=True, position="inside"),
#         itemstyle_opts=opts.ItemStyleOpts(border_color="#fff", border_width=1),
#     )
#     .set_global_opts(title_opts=opts.TitleOpts(title="漏斗图2", subtitle="纯属虚构2"))
# #     .render_notebook()
# )

# (
#     Grid(init_opts=opts.InitOpts())
#     .add(
#         funnel1, grid_opts=opts.GridOpts(pos_right="58%"), is_control_axis_index=True
#     )
#     .add(funnel2, grid_opts=opts.GridOpts(pos_left="58%"), is_control_axis_index=True)
#     .render_notebook()
# )


# In[227]:


fig1,axs=plt.subplots(1,3,figsize=(16,6))
age = [
    [311392,155356,202144,191932,165705,162209],
    [296856,157302,191324,198409,169512,165119],
    [280033,163118,184775,204723,163449,168889]
]
years = [2007, 2008, 2009]
columns = ['0-19', '20-29', '30-39', '40-49', '50-59', '60+']
explode = (0.1, 0, 0, 0, 0, 0.1) 
color_list = ['b','g','r']
for i, ax in enumerate(axs):
    labels = columns
    sizes = age[i]
    ax.pie(sizes,  labels=labels, explode=explode, autopct='%1.1f%%',
        shadow=False, startangle=90)
    ax.set_title(f'Year {years[i]} Age Structure')
fig1.show()
fig1.savefig('age_structure.png')


# ## Bed regression

# In[284]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
beds = bed_num
elders = elder_num

poly_features_1 = PolynomialFeatures(degree = 1)
bed_regr = LinearRegression(poly_features_1)
poly_features_2 = PolynomialFeatures(degree = 1)
elders_regr = LinearRegression(poly_features_2)

bed_regr.fit(np.arange(8).reshape(-1,1), beds)
elders_regr.fit(np.arange(8).reshape(-1,1), elders)


# In[285]:


bed_20 = bed_regr.predict(np.arange(20).reshape(-1,1))
elder_20 = elders_regr.predict(np.arange(20).reshape(-1,1))


# In[286]:


x_data = ["2019年", "2020年", "2021年", "2022年", "2023年", "2024年", "2025年", "2026年", "2027年", "2028年"]

(
    Line()
    .add_xaxis(xaxis_data=x_data)
    .add_yaxis(
        series_name="床位供应量",
        stack="总量",
        y_axis=list(map(int, (bed_20[9:19]*100).tolist())),
        areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
        label_opts=opts.LabelOpts(is_show=True),
    )
    .add_yaxis(
        series_name="床位需求量",
        stack="总量",
        y_axis=list(map(int, (elder_20[9:19]*10).tolist())),
        areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
        label_opts=opts.LabelOpts(is_show=True, position="top"),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="未来10年床位需求量与供应量预测"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
            axislabel_opts=opts.LabelOpts(formatter="{value} 万人")
        ),
        legend_opts=opts.LegendOpts(),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
    )
    .render_notebook()
)


# In[310]:


fig2,axs=plt.subplots(1,2,figsize=(14,4))
x_data = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]
y_scatter = [have, need]
y_line = [[have[0], have[-1]], [need[0], need[-1]]]
color_list = ['b','g','r']
title_list = [f'Beds Own in Feture',f'Beds Need in Feture']

for i, ax in enumerate(axs):
    x = x_data
    yc = y_scatter[i]
    yl = y_line[i]
    sns.scatterplot(x, yc, ax = ax, color='r')
    sns.lineplot([x[0], x[-1]], yl, ax = ax)
    
    ax.set_title(title_list[i])
    ax.set_ylabel('Number of beds (10k)')
    ax.set_xlabel('year')
fig2.show()
fig2.savefig('beds_predict.png')


# In[319]:


fig3,axs=plt.subplots(1,3,figsize=(16,6))
x_data = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]
i_cent = np.array([3.1, 3.05, 2.9, 2.8, 2.75, 2.6, 2.5, 2.4, 2.2, 2.01])
c_cent = np.array([4.01, 4.3, 4.6, 4.75, 5.24, 5.4, 5.6, 5.7, 5.8, 6])
f_cent = 100 - i_cent - c_cent
y_data = [i_cent, c_cent, f_cent]
color_list = ['b','g','r']
title_list = ['Proportion of institutional pension',f'Proportion of community pension',f'Proportion of Family pension']

for i, ax in enumerate(axs):
    x = x_data
    y = y_data[i]
    sns.scatterplot(x, y, ax = ax, color=color_list[i])
#     sns.lineplot([x[0], x[-1]], [y[0], y[-1]], ax = ax)
    sns.lineplot(x, y, ax = ax, color=color_list[i])
    ax.set_title(title_list[i])
    ax.set_ylabel('Proportion %')
    ax.set_xlabel('year')
fig3.show()
fig3.savefig('category_predict.png')


# In[346]:


fig4,axs=plt.subplots(1,3,figsize=(16,6))
y_data = [df_2017.social_services.values, df_2017.capital_construction.values, df_2017.assets.values]
x_data = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
color_list = ['b','g','r']
title_list = ['Social service expenses', 'Capital construction completed investment', 'Original price of fixed assets']

for i, ax in enumerate(axs):
    x = x_data
    y = y_data[i]
    sns.barplot(x, y, ax = ax)
#     ax = ax.twinx()
#     ax.lineplot(x, y, ax = ax)
    ax.set_title(title_list[i])
    ax.set_ylabel('expenditure (Billion)')
    ax.set_xlabel('year')
fig4.show()
fig4.savefig('expenditure_of_social_services.png')


# In[326]:


df_2017


# In[330]:


x


# In[331]:


y


# In[347]:


birth_rate = [21.06,19.68,18.24,18.09,17.7,17.12,16.98,16.57,15.64,14.64,14.03,13.38,12.86,12.41,12.29,12.4,12.09,12.1,12.14]
death_rate = [6.67,6.7,6.64,6.64,6.49,6.57,6.56,6.51,6.5,6.46,6.45,6.43,6.41,6.4,6.42,6.51,6.81,6.93,7.06]
natural_growth_rate = [14.39,12.98,11.6,11.45,11.21,10.55,10.42,10.06,9.14,8.18,7.58,6.95,6.45,6.01,5.87,5.89,5.28,5.17,5.08]
total_population = [114333,115823,117171,118517,119850,121121,122389,123626,124761,125786,126743,127627,128453,129227,129988,130756,131448,132129,132802
]
birth_rate = np.array(birth_rate)
death_rate = np.array(death_rate)
natural_growth_rate = np.array(natural_growth_rate)
total_population = np.array(total_population)

birth = total_population*birth_rate
death = total_population*death_rate
growth = total_population*natural_growth_rate


# In[357]:


fig5,axs=plt.subplots(1,3,figsize=(16,6))
y_data = [birth_rate[8:], death_rate[8:], natural_growth_rate[8:]]

x_data = list(range(1990,2009))[8:]

color_list = ['b','g','r']
title_list = ['birth rate', 'death rate', 'natural growth rate']

for i, ax in enumerate(axs):
    x = x_data
    y = y_data[i]
    sns.scatterplot(x, y, ax = ax, color=color_list[i])
    sns.lineplot(x, y, ax = ax, color=color_list[i])
    ax.set_title(title_list[i])
    ax.set_ylabel('percentage %')
    ax.set_xlabel('year')
    
#     plt.xticks(rotation=30)
fig5.show()
fig5.savefig('population.png')


# In[ ]:





# In[365]:


fig1,axs=plt.subplots(1,3,figsize=(16,6))
pool = [
    [1480.2, 5207.2, 529.1],
    [1261,4045,466.9],
    [1007,3519,455]
]
years = [2016, 2017, 2018]
columns = ['city_MLA', 'countryside_MLA', 'countryside_SP']
color_list = ['b','g','r']
for i, ax in enumerate(axs):
    labels = columns
    sizes = pool[i]
    ax.pie(sizes,  labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
    ax.set_title(f'Year {years[i]} Poor Population')
fig1.show()
fig1.savefig('Poor_Population.png')


# In[ ]:





# In[368]:


from pandas_datareader import data

df = data.get_data_yahoo("SPY")
df['return'] = df['Adj Close'].pct_change().fillna(0)


# In[369]:


df.head()


# In[371]:


import pandas_montecarlo
mc = df['return'].montecarlo(sims=10, bust=-0.1, goal=1)
mc.plot(title="SPY Returns Monte Carlo Simulations")  # optional: , figsize=(x, y)


# In[375]:


mc = df_std['beds'].montecarlo(sims=10, bust=-0.1, goal=1)
mc.plot(title='Expected Return', figsize=(10, 6))


# In[397]:


from pyecharts import options as opts
from pyecharts.charts import Graph

nodes = [
    {"name": "养老服务", "symbolSize": 100},
    {"name": "政府补贴", "symbolSize": 30},
    {"name": "社会捐赠", "symbolSize": 30},
    {"name": "政策支持", "symbolSize": 40},
    {"name": "养老就业", "symbolSize": 50},
    {"name": "社会福利", "symbolSize": 60},
    {"name": "持续发展", "symbolSize": 30},
]
links = []
for i in nodes:
    for j in nodes:
        links.append({"source": i.get("name"), "target": j.get("name")})
(
    Graph()
    .add("", nodes, links, repulsion=10000)
    .set_global_opts(title_opts=opts.TitleOpts(title="养老服务关系图",  pos_left='43%'))
    .render_notebook()
)


# In[ ]:





# In[410]:


import math
from typing import Union

import pyecharts.options as opts
from pyecharts.charts import Surface3D


def float_range(start: int, end: int, step: Union[int, float], round_number: int = 2):
    temp = []
    while True:
        if start < end:
            temp.append(round(start, round_number))
            start += step
        else:
            break
    return temp


def surface3d_data():
    for t0 in float_range(-3, 3, 0.05):
        y = t0
        for t1 in float_range(-3, 3, 0.05):
            x = t1
            z = (math.sin(x)+math.cos(y))/2
            yield [x, y, z]


(
    Surface3D(init_opts=opts.InitOpts())
    .add(
        series_name="",
        shading="color",
        data=list(surface3d_data()),
        xaxis3d_opts=opts.Axis3DOpts(type_="value"),
        yaxis3d_opts=opts.Axis3DOpts(type_="value"),
        grid3d_opts=opts.Grid3DOpts(width=100, height=80, depth=100),
    )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            dimension=2,
            max_=1,
            min_=-1,
            range_color=[
                "#313695",
                "#4575b4",
                "#74add1",
                "#abd9e9",
                "#e0f3f8",
                "#ffffbf",
                "#fee090",
                "#fdae61",
                "#f46d43",
                "#d73027",
                "#a50026",
            ],
        ),
    title_opts=opts.TitleOpts(title="养老业与经济发展模拟图",  pos_left='38%', pos_top='10%')
    )
    .render_notebook()
)


# In[ ]:




