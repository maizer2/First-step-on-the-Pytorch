import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

# 입력 데이터
num_data = 1000

# 반복 횟수
num_epoch = 10000

#init.normal_함수는 빈 텐서에 정규분포를 따르는 값을 랜덤하게 넣어줍니다.
# 이를 벡터공간에 있는 랜덤백터 즉 노이즈라고 말합니다.
## 정규분포는 입력해준 파라미터값에 따라 만들어집니다.
### normal_(tensor, mean=0.0, std=1.0)
noise = init.normal_(torch.FloatTensor(num_data, 1)) #num_data 행, 1열

# -15 ~ 15까지 랜점하게 값을 채운다
x = init.uniform_(torch.Tensor(num_data, 1), -15, 15) #num_data 행, 1열

# y = x^{2} + 3
y = (x**2) + 3

# pytorch의 브로드캐스팅 기능으로 인해 각 행 + y가 된다.
y_noise = y + noise # 결과는 기존의 y와 같다.

model = nn.Sequential(
        nn.Linear(1, 6),
        nn.ReLU(),
        nn.Linear(6, 10),
        nn.ReLU(),
        nn.Linear(10, 6),
        nn.ReLU(),
        nn.Linear(6, 1),
        )

loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr= 0.0002)

loss_array = []

for i in range(num_epoch):
    optimizer.zero_grad()
    output = model(x)
    loss = loss_func(output, y_noise)
    loss.backward()
    optimizer.step()

    loss_array.append(loss)

import matplotlib.pyplot as plt

plt.plot(loss_array)
plt.savefig("loss_array.png")
