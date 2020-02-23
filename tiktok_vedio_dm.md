
## Travis Project: 基于数据挖掘的tik tok商用广告分类


在这个文件中，部分的模版代码已经写好，但你还需要补充一些功能来让项目成功运行。
除非有明确要求，你无须修改任何我已给出的代码
。
以**编程练习**开始的标题表示接下来的内容中有需要你必须实现的功能。每一部分都会有指导，需要实现的部分也会在注释中以**TODO**标出。
请仔细阅读提示！！！

- **Your Task**：一般来说，为了吸引观众的注意力，广告视频的长度、音频、文本位置和画面会有与众不同之处。我将通过对tik tok平台上视频的时长、声音频谱、视频光谱、文字分布和画面变化等特征，构建一套商用广告视频识别系统来快速区分出投稿视频中的商用广告。
- **项目步骤**：    
                        1. 数据的探索与问题分析；
                        2. 清洗数据；
                            a.处理缺失值；
                            b.标签转换；
                            c.查看重复；
                        3. 特征工程；
                            a.特征选择；
                            b.特征生成；
                            c.特征分箱；
                        4. 选择模型进行交叉验证和网格搜索；
                        5. 模型的集成；
                        6. 模型评价的深入思考；
                        7. 进一步思考该项目。                     

>**提示：**Code 和 Markdown 区域可通过 **Shift + Enter** 快捷键运行。此外，Markdown可以通过双击进入编辑模式。
>**注意：**如果有未安装的库，请使用 `pip install`,`conda install`命令进行安装

---
## Thinking  before project begin

> 在本项目开始之前，请仔细阅读下面这段话，带着这段话继续下面的项目。

### To judge whether it is commercial advertising or not through short video feature data.T
    
> 数据收集了129685份视频的信息，共230个特征(维度较高），涵盖视频的时长、声音频谱、视频光谱、文字分布和画面变化等方面。
    
**问题1：** 你认为广告视频和一般视频在上述方面会有怎样的区别？ 

**回答：** 

**问题2：** 这里我采取的是人为指定特征的方法，你知道什么方法可以直接从视频数据中学习出特征么？ 

**回答：** 

> 需要用机器学习来帮助判断视频的类型。理论上得到的有效信息越多，越容易提升模型的预测效果。但是 ***更多的特征***  也意味着不是所有视频都会具有此情况，因此导致特征矩阵稀疏。
    
**问题3：** 如何处理这种原因导致的稀疏（即如何处理缺失值）？ 

**回答：** 

> 需要人为指定特征来构建模型，一般而言这是极其困难的。虽然我们学习前人的方法获得了这些特征，你仍可通过进一步的特征工程提升预测效果。
  
**问题4：** 对此你有什么想法？ 

**回答：** 

---
## Step 1：EDA
阅读《数据集和变量说明.pdf》文档，简单了解数据，并描述数据基本特征。
数据文件如下：
- commercial_vedio_data.csv


```python
# 请不要修改此格代码
# 导入依赖库
from matplotlib import pyplot as plt
%matplotlib inline
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
#加载数据
data = pd.read_csv('commercial_vedio_data.csv',index_col=0)
col_name= data.columns[:-2]
label_name = data.columns[-1]
#查看数据的标签
print('训练集的标签：{}\n'.format(label_name))
#查看数据的特征
print('训练集的特征：{}\n'.format(col_name))
#查看数据的shape
print('训练集的形状：{}'.format(data.shape))
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-1-bbd0d1b29e9f> in <module>
          8 warnings.filterwarnings("ignore")
          9 #加载数据
    ---> 10 data = pd.read_csv('commercial_vedio_data.csv',index_col=0)
         11 col_name= data.columns[:-2]
         12 label_name = data.columns[-1]


    /srv/conda/envs/notebook/lib/python3.6/site-packages/pandas/io/parsers.py in parser_f(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)
        683         )
        684 
    --> 685         return _read(filepath_or_buffer, kwds)
        686 
        687     parser_f.__name__ = name


    /srv/conda/envs/notebook/lib/python3.6/site-packages/pandas/io/parsers.py in _read(filepath_or_buffer, kwds)
        455 
        456     # Create the parser.
    --> 457     parser = TextFileReader(fp_or_buf, **kwds)
        458 
        459     if chunksize or iterator:


    /srv/conda/envs/notebook/lib/python3.6/site-packages/pandas/io/parsers.py in __init__(self, f, engine, **kwds)
        893             self.options["has_index_names"] = kwds["has_index_names"]
        894 
    --> 895         self._make_engine(self.engine)
        896 
        897     def close(self):


    /srv/conda/envs/notebook/lib/python3.6/site-packages/pandas/io/parsers.py in _make_engine(self, engine)
       1133     def _make_engine(self, engine="c"):
       1134         if engine == "c":
    -> 1135             self._engine = CParserWrapper(self.f, **self.options)
       1136         else:
       1137             if engine == "python":


    /srv/conda/envs/notebook/lib/python3.6/site-packages/pandas/io/parsers.py in __init__(self, src, **kwds)
       1915         kwds["usecols"] = self.usecols
       1916 
    -> 1917         self._reader = parsers.TextReader(src, **kwds)
       1918         self.unnamed_cols = self._reader.unnamed_cols
       1919 


    pandas/_libs/parsers.pyx in pandas._libs.parsers.TextReader.__cinit__()


    pandas/_libs/parsers.pyx in pandas._libs.parsers.TextReader._setup_parser_source()


    FileNotFoundError: [Errno 2] File b'commercial_vedio_data.csv' does not exist: b'commercial_vedio_data.csv'


### 编程练习
打印数据集的前5行数据


```python
# TODO：打印data的前五行数据

```

### 编程练习
查看data数据集中标签('lables')的分布


```python
# TODO：描述数据中标签的分布

```


```python
# TODO：画出标签分布的直方图

```

### 编程练习
查看data数据集特征的的分布，对于describe善用转置可展示更多特征<br>
真正了解你的数据，而非走形式


```python
# TODO：描述数据中特征的分布(表1)

```


```python
# TODO：描述数据中特征的分布(表2)

```


```python
# TODO：描述数据中特征的分布(表3)

```


```python
# TODO：描述数据中特征的分布(表4)

```

---
## 步骤二：Clean data

在这一步中将开始清洗数据

通过观察数据发现可：

- 数据需要查看是否存在缺失值；
- 数据需要查看是否存在重复样本；
- 虽然不同样本的相同特征取值变化不大，但特征间数值差距大；
- 数据的标签需要变化。


### 编程练习
统计数据中的缺失值的数量
- 统计数据集有多少行有缺失
- 统计数据集有多少列有缺失


```python
# TODO：统计数据集中有多少行、列存在缺失

```

### 编程练习
统计数据中的重复样本的数量
- 统计数据集有多少行重复


```python
# TODO：统计数据集中有多少行重复

```

### 思考
从上面可以看出数据的缺失值很多。结合文档中介绍的我们得到这些数据的方式，你觉得应如何处理缺失值？<br>
下面给出一种思路：


```python
# TODO：将所有缺失值以0填补

```


```python
# TODO：请给出原因为何用0填充
```


```python
# TODO：将缺失值用统计平均数填充（提升）
```

### 注意
从上面我们可以看出特征间的尺度差距较大，注意在建模前结合模型要求进行适当的处理。


```python
# TODO：请给出你的见解，为何特征间尺度差距较大？
1：
2：
3：
```

### 标签数据转换
将labels的值中的“-1”转换为0


```python
# TODO：将标签中的‘-1’转换为‘0’

```

### 分类特征重编码

从上面的**数据探索**中的表中，我们可以看到有几个特征是无序的分类特征。因为无序特征各属性之间不能比较大小，通常情况下，要求无序特征（称为无序类别变量）被转换。无序转换类别变量的一种流行的方法是使用**独热编码**方案。独热编码为每一个无序分类特征的每一个可能的类别创建一个_“虚拟”_变量。例如，假设`someFeature`有三个可能的取值`A`，`B`或者`C`。我们将把这个特征编码成`someFeature_A`, `someFeature_B`和`someFeature_C`.

| someFeature | someFeature_A | someFeature_B | someFeature_C |
| :-: | :-: | :-: | :-: |
|  A  | 1 | 0 | 0 |
|  B  | 0 | 1 | 0 |
|  C  | 0 | 0 | 1 |

 - 使用[`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies)对`'features_raw'`数据来施加一个独热编码。

结合你对数据的了解，你认为有没有特征需要进行one-hot编码？


```python
#Answer：
```


```python
# TODO：对你认为需要的特征进行one-hot编码
```

## Step 2：Feature Engineering

- 特征过滤
- 特征生成
- 特征分箱

可以发现，可综合多种方法进行特征工程。
一是使用某些方法生成新的特征纳入模型进行预测；
二是通过某些方法进行特征过滤，减少纳入模型的特征数量；
三是对连续特征进行特征分箱，离散特征进行特征组合。
在此给出简单的特征工程示例。

### 分离特征和标签


```python
# TODO：分离特征和标签

```

### 划分训练集和测试集
为避免测试集在特征工程步骤中被引入训练集信息，在进行特征工程前划分训练集和验证集。


```python
# TODO：将数据集划分为训练集和测试集

```

### 使用随机森林进行特征选择
你也可以使用你所知道的其他方法进行特征选择

我们选择一个有 `'feature_importance_'` 属性的scikit学习分类器（例如随机森林）。         
`'feature_importance_'` 属性是对特征的重要性排序的函数。在后面我们将使用这个分类器拟合训练集数据并使用这个属性来对特征的重要程度进行排序。


```python
## TODO：使用随机森林进行特征选择，首先在训练集拟合随机森林模型

```


```python
## TODO：将特征的重要性程度进行排序

```


```python
# 保留训练集和测试集中对于分类最重要的前n个特征（自主考虑保留多少个特征）
# 以保留对于分类最重要的五十个特征为例，具体多少可自己斟酌
imp = np.argsort(rf.feature_importances_)[::-1]
imp_slct = imp[:50]
X_train_slct=pd.DataFrame(X_train).iloc[:,imp_slct]
X_test_slct=X_test.iloc[:,imp_slct]
```


```python
## TODO：尝试生成特征排序后的热力图
```

###  使用PCA进行特征生成
你也可以用你所知道的其他方法进行此步 （提示：考虑T-NSE做可视化）

PCA除了用于降维外，还可用于特征生成，即将选择出的主成分与原数据合并，一般会提升原数据的预测能力。


```python
# TODO：对训练集使用PCA生成新特征,首先查看各主成分解释方差比例

```


```python
# 对训练集使用PCA生成新特征,根据累计贡献率，保留前5个主成分

```


```python
# TODO：对测试集进行相同的操作，注意测试集上直接使用pca中的transform函数

```

###  使用决策树进行特征分箱

你也可以用你所知道的其他方法进行此步

特征分箱是常用的一种特征工程方法，它将连续变量离散化。合理的离散化将提升数据的预测能力。实现特征分箱的方法很多：

- 简单地将数据等距分成n份
- 简单地将数据等频分成n份
- 根据卡方统计量优化分箱
- 根据决策树优化分箱
- 根据其他算法优化分箱

一般还需根据WOE和IV评价分箱效果，在此不做进一步拓展了。

接下来的两个函数是决策树优化分箱的示例。你也可以尝试将他们改装成类，实现更方便的调用。


```python
# TODO：将训练集及其标签、测试集及其标签合并，以便函数使用

```


```python
'''
cut_bin参数说明
    df：训练集名称，如train
    label：标签名称，如'labels'
    max_depth：决策树最大深度
    p：最小叶子节点数与样本量的比例
cut_bin输出说明
    df_bin：分箱后的数据集
    dict_bin：储存分箱参数的字典，包含了用以评价分箱的WOE和IV
'''
'''
cut_test_bin参数说明
    df：测试集名称，如test
    label：标签名称，如'labels'
    train_dict_bin：训练集分箱生成的储存参数的字典
cut_test_bin输出说明
    df_bin：分箱后的数据集
    dict_bin：储存分箱参数的字典，包含了用以评价分箱的WOE和IV
    '''
```


```python
#  cut_bin对训练集进行分箱
#  cut_test_bin对测试集进行分箱



```


```python
from sklearn.tree import DecisionTreeClassifier
# TODO：调用函数完成分箱

```

### 分重新离特征和标签


```python
# TODO：再次分离特征和标签，注意训练集和测试集都要进行

```


```python
#Answer：为何要做特征标签重新分离，请给出个人看法：

```

---
## Step 4：Select models for cross validation and grid search

请选择合适的模型以及评估方式，使用交叉验证和网格搜索建立模型，并选择合适的参数,打印出交叉验证的结果。

- clf_model是分类模型。
- 注意模型对数据规整的需求。


```python
## TODO：模型选择、交叉验证、网格搜索
clf_model = None

# 这里演示最简单的模型更多的模型选择和参数调整将由你自己完成
# 分类模型
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_features=16,max_depth=12,n_estimators=2048,n_jobs=-1,random_state=0)
rf.fit(X_train_bin,y_train)
```


```python
#Answer:请上网完成随机森林参数调整调研并整理常用高sota参数具体值，并给出自己的调参过程见解（至少3次调参）


#示例：
#Commit X
* Dropout_par = 0.2 (everywhere)
* learning_rate_lgb = 0.02
* learning_rate_xgb = 0.0002
* learning_rate_nn = 4e-5
* epochs_nn=25
* weights_models = {'lbg': 0.6, 'xgb': 0.2, 'nn': 0.2}

```

### Deep thinking

- 对于分类预测稳定性问题，我们使用accuracy、混淆矩阵和AUC指标来评估模型。

>**注意：** 除此以外，还可以设计一个随机猜测函数作为baseline来进一步查看我们的模型相对于随机猜测是否有很大的优势。



```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score
# 分类模型测试集上效果
# auc和混淆矩阵评估
y_train_pred_clf = rf.predict_proba(X_train_bin)
y_train_pred = rf.predict(X_train_bin)
y_test_pred_clf = rf.predict_proba(X_test_bin)
y_test_pred = rf.predict(X_test_bin)
# 评估训练集效果，直观判断是否过拟合


# 评估测试集效果

# 随机猜测函数对比

```

---
## Step 5：Model integration
尝试使用stacking等方法对步骤四生成的不同学习器进行集成。

注意评价集成后的模型。


```python
## TODO:模型集成
ensemble_model_clf = None
```


```python
## TODO:请写出你的模型集成构思思路

```

---
## Step 6: in-depth thinking on model evaluation<br>
<br>
找到你认为最好的模型后，设定不同的分类阈值(threshold)仍会导致模型的实际应用效果有所不同，这无疑会影响模型的实际使用效果。
ROC曲线是以模型在不同阈值下的真阳性率为纵轴，假阳性率为横轴绘制而成，它不仅可以不依靠阈值评价模型的预测效果。


```python
## 计算各阈值下假阳性率、真阳性率和AUC
from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold = roc_curve(y_test,y_test_pred_clf[:,1])
roc_auc = auc(fpr,tpr)
```


```python
## TODO:假阳性率为横坐标，真阳性率为纵坐标做曲线

```


```python
## TODO:请画出混淆矩阵图

```

虽然前述建议使用ACC和AUC评价模型，但是针对想要找出的tik tok商用视频这一目的，你认为有其他更合适的指标？结合混淆矩阵予以说明。


```python
## TODO:
```

---
## Step 7: think about the project further（该题我没做）

你可以将整个流程调整为你认为更合理的方式；

尝试一下综合应用特征选择和特征生成，选择最适合本问题的方法；

可以在特征选择，**之后**再生成新的特征。特征选择可以采取包裹法、过滤法或其他嵌入法进行。


```python
## TODO：特征选择+特征生成

```
