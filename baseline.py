#!/usr/bin/env python
# coding: utf-8

# ## 赛事介绍
# 实时对战游戏是人工智能研究领域的一个热点。由于游戏复杂性、部分可观察和动态实时变化战局等游戏特点使得研究变得比较困难。我们可以在选择英雄阶段预测胜负概率，也可以在比赛期间根据比赛实时数据进行建模。那么我们英雄联盟对局进行期间，能知道自己的胜率吗？
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/9739d3ca3cef4e32989a541af450a9556e91bf89a4e946e0a856cc2424321638)
# 
# 
# ## 赛事任务
# 比赛数据使用了英雄联盟玩家的实时游戏数据，记录下用户在游戏中对局数据（如击杀数、住物理伤害）。希望参赛选手能从数据集中挖掘出数据的规律，并预测玩家在本局游戏中的输赢情况。
# 
# 赛题训练集案例如下：
# - 训练集18万数据；
# - 测试集2万条数据；
# 
# ```plain
# import pandas as pd
# import numpy as np
# 
# train = pd.read_csv('train.csv.zip')
# ```
# 
# 对于数据集中每一行为一个玩家的游戏数据，数据字段如下所示：
# 
# * id：玩家记录id
# * win：是否胜利，标签变量
# * kills：击杀次数
# * deaths：死亡次数
# * assists：助攻次数
# * largestkillingspree：最大 killing spree（游戏术语，意味大杀特杀。当你连续杀死三个对方英雄而中途没有死亡时）
# * largestmultikill：最大mult ikill（游戏术语，短时间内多重击杀）
# * longesttimespentliving：最长存活时间
# * doublekills：doublekills次数
# * triplekills：doublekills次数
# * quadrakills：quadrakills次数
# * pentakills：pentakills次数
# * totdmgdealt：总伤害
# * magicdmgdealt：魔法伤害
# * physicaldmgdealt：物理伤害
# * truedmgdealt：真实伤害
# * largestcrit：最大暴击伤害
# * totdmgtochamp：对对方玩家的伤害
# * magicdmgtochamp：对对方玩家的魔法伤害
# * physdmgtochamp：对对方玩家的物理伤害
# * truedmgtochamp：对对方玩家的真实伤害
# * totheal：治疗量
# * totunitshealed：痊愈的总单位
# * dmgtoturrets：对炮塔的伤害
# * timecc：法控时间
# * totdmgtaken：承受的伤害
# * magicdmgtaken：承受的魔法伤害
# * physdmgtaken：承受的物理伤害
# * truedmgtaken：承受的真实伤害
# * wardsplaced：侦查守卫放置次数
# * wardskilled：侦查守卫摧毁次数
# * firstblood：是否为firstblood
# 测试集中label字段win为空，需要选手预测。
# 
# ##  评审规则
# 
# 1. 数据说明
# 
# 选手需要提交测试集队伍排名预测，具体的提交格式如下：
# 
# ```plain
# win
# 0
# 1
# 1
# 0
# ```
# 
#  2. 评估指标
# 
# 本次竞赛的使用准确率进行评分，数值越高精度越高，评估代码参考：
# 
# ```
# from sklearn.metrics import accuracy_score
# y_pred = [0, 2, 1, 3]
# y_true = [0, 1, 2, 3]
# accuracy_score(y_true, y_pred)
# ```

# ## Baseline使用指导
# 1、点击‘fork按钮’，出现‘fork项目’弹窗         
# 2、点击‘创建按钮’ ，出现‘运行项目’弹窗    
# 3、点击‘运行项目’，自动跳转至新页面      
# 4、点击‘启动环境’ ，出现‘选择运行环境’弹窗     
# 5、选择运行环境（启动项目需要时间，请耐心等待），出现‘环境启动成功’弹窗，点击确定        
# 6、点击进入环境，即可进入notebook环境      
# 7、鼠标移至下方每个代码块内（代码块左侧边框会变成浅蓝色），再依次点击每个代码块左上角的‘三角形运行按钮’，待一个模块运行完以后再运行下一个模块，直至全部运行完成  
# ![](https://ai-studio-static-online.cdn.bcebos.com/226c72f88f5b4e9d8a55e59129e4c79770aa200f10ef413ca1420ae7d273bc88)  
# ![](https://ai-studio-static-online.cdn.bcebos.com/866a22a341d64166aaf9a8a3abee09b5a6e2d0cba1c649bb8bdef6b2ad7955f1)  
# 8、下载页面左侧submission.zip压缩包  
# ![](https://ai-studio-static-online.cdn.bcebos.com/b7f3076301e34462abaf2013dcdbf10a5dcbfe287d5845f1869493e578391f7a)  
# 9、在比赛页提交submission.zip压缩包，等待系统评测结束后，即可登榜！    
# ![](https://ai-studio-static-online.cdn.bcebos.com/95a9fc4140144a2d8a76258f7a536751c5c6969994154d3da71c80e23757c6c8)  
# 10、点击页面左侧‘版本-生成新版本’  
# ![](https://ai-studio-static-online.cdn.bcebos.com/e62d0f5ca1454f7485c3eb1351823315480356cafad143c2ab2ff065f95104fe)  
# 11、填写‘版本名称’，点击‘生成版本按钮’，即可在个人主页查看到该项目（可选择公开此项目哦）  

# In[1]:


import pandas as pd
import paddle
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('pylab', 'inline')
import seaborn as sns

train_df = pd.read_csv('train.csv.zip')
test_df = pd.read_csv('test.csv.zip')

train_df = train_df.drop(['id', 'timecc'], axis=1)
test_df = test_df.drop(['id', 'timecc'], axis=1)


import os
import sys

# 创建保存图片的目录
if not os.path.exists('plots'):
    os.makedirs('plots')

# 重定向print输出到文件
class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()

# ## 数据分析

# In[3]:


print("Missing values ratio:")
print(train_df.isnull().mean(0))


# In[4]:


plt.figure()
train_df['win'].value_counts().plot(kind='bar')
plt.title('Win Distribution')
plt.savefig('plots/win_distribution.png')
plt.close()


# In[5]:


plt.figure()
sns.distplot(train_df['kills'])
plt.title('Kills Distribution')
plt.savefig('plots/kills_distribution.png')
plt.close()


# In[5]:


plt.figure()
sns.distplot(train_df['deaths'])
plt.title('Deaths Distribution')
plt.savefig('plots/deaths_distribution.png')
plt.close()


# In[6]:


plt.figure()
sns.boxplot(y='kills', x='win', data=train_df)
plt.title('Kills vs Win Boxplot')
plt.savefig('plots/kills_win_boxplot.png')
plt.close()


# In[7]:


plt.figure()
plt.scatter(train_df['kills'], train_df['deaths'])
plt.xlabel('kills')
plt.ylabel('deaths')
plt.title('Kills vs Deaths Scatter')
plt.savefig('plots/kills_deaths_scatter.png')
plt.close()


# In[8]:


for col in train_df.columns[1:]:
    train_df[col] /= train_df[col].max()
    test_df[col] /= test_df[col].max()


# ## 搭建模型

# In[9]:


class Classifier(paddle.nn.Layer):
    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Classifier, self).__init__()
        
        self.fc1 = paddle.nn.Linear(in_features=29, out_features=40)
        self.fc2 = paddle.nn.Linear(in_features=40, out_features=1)
        self.relu = paddle.nn.ReLU()
    
    # 网络的前向计算
    def forward(self, inputs):
        x = self.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x


# In[10]:


model = Classifier()
model.train()
opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
loss_fn = paddle.nn.BCEWithLogitsLoss()


# In[11]:


EPOCH_NUM = 2   # 设置外层循环次数
BATCH_SIZE = 100  # 设置batch大小
training_data = train_df.iloc[:-1000,].values.astype(np.float32)
val_data = train_df.iloc[-1000:, ].values.astype(np.float32)

# 定义外层循环
for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    
    np.random.shuffle(training_data)
    
    # 将训练数据进行拆分，每个batch包含10条数据
    mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
    
    # 定义内层循环
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, 1:]) # 获得当前批次训练数据
        y = np.array(mini_batch[:, :1]) # 获得当前批次训练标签
        
        # 将numpy数据转为飞桨动态图tensor的格式
        features = paddle.to_tensor(x)
        y = paddle.to_tensor(y)
        
        # 前向计算
        predicts = model(features)
        
        # 计算损失
        loss = loss_fn(predicts, y, )
        avg_loss = paddle.mean(loss)
        if iter_id%200==0:
            acc = (predicts > 0).astype(int).flatten() == y.flatten().astype(int)
            acc = acc.astype(float).mean()

            print("epoch: {}, iter: {}, loss is: {}, acc is {}".format(epoch_id, iter_id, avg_loss.numpy(), acc.numpy()))
        
        # 反向传播，计算每层参数的梯度值
        avg_loss.backward()
        # 更新参数，根据设置好的学习率迭代一步
        opt.step()
        # 清空梯度变量，以备下一轮计算
        opt.clear_grad()


# In[12]:


model.eval()
test_data = paddle.to_tensor(test_df.values.astype(np.float32))
test_predict = model(test_data)
test_predict = (test_predict > 0).astype(int).flatten()


# In[13]:


pd.DataFrame({'win':
              test_predict.numpy()
             }).to_csv('submission.csv', index=None)

import os
os.system('zip submission.zip submission.csv')
# get_ipython().system('zip submission.zip submission.csv')


# ## 总结与上分点
# 
# 1. 原始赛题字段存在关联，可以进一步提取交叉特征。
# 2. 模型训练过程中可以加入验证集验证过程。

# In[ ]:




