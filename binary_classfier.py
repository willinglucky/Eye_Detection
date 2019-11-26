import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 生成数据
# 分别生成2组各100个数据点，增加正态噪声，后标记以y0=0 y1=1两类标签，最后cat连接到一起
n_data = torch.ones(100, 2)
# torch.normal(means, std=1.0, out=None)
x0 = torch.normal(2 * n_data, 1)  # 以tensor的形式给出输出tensor各元素的均值，共享标准差
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # 组装（连接）
y = torch.cat((y0, y1), 0).type(torch.LongTensor)

# 置入Variable中
x, y = Variable(x), Variable(y)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.012)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()
plt.show()

for t in range(100):
    out = net(x)
    loss = loss_func(out, y)  # loss是定义为神经网络的输出与样本标签y的差别，故取softmax前的值

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        # torch.max既返回某个维度上的最大值，同时返回该最大值的索引值
        prediction = torch.max(F.softmax(out), 1)[1]  # 在第1维度取最大值并返回索引值
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accu=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()