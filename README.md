# cifar-10
## 实验内容
### step 1: 检查初始Loss
![initial_loss](https://user-images.githubusercontent.com/42258146/141669313-48b5990a-e425-47e2-8904-7988b950d654.png)

正常初始Loss为2.4左右，该模型初始Loss为2.32，处于正常范围。
### step 2: 在小样本上拟合
![small_sample](https://user-images.githubusercontent.com/42258146/141669366-7eb44337-6250-4bfb-a23b-5ec472e9a04c.png)
取batch_size为256的10个mini_batch, 在该数据上拟合，567次迭代后准确率达到0.954，说明该模型对于cifar-10数据集是有效的。
### step 3: 找到使梯度下降得最快的学习率
选取learning_rate为1e-1, 1e-2, 1e-3, 1e-4, 对所有batch迭代一个epoch，结果如下：
![learning_rate](https://user-images.githubusercontent.com/42258146/141669501-96fbca5c-2dd4-4365-ac3b-a52b074d034f.png)
可以看到，当lr为0.1时，Loss下降的最快，但是波动比较大，而lr为0.01时，Loss稳定下降。当lr为1e-3和1e-4时，Loss下降的十分缓慢。
### step 4: 根据不同参数，训练1-5个epoch
由上一步我们可以看到，比较合适的学习率为0.1或0.01，因此我们设置学习率为这两个参数，再组合weight_decay为1e-4, 1e-5, 0训练5个epoch，结果如下:
| learning_rate | weight_decay | acc |
| :-----: | :----: | :----: |
| 1e-1 | 1e-4 | 70.4 |
| 1e-1 | 1e-5 | 71.1 |
| 1e-1 | 0 | 71.5 |
| 1e-2 | 1e-4 | 44.3 |
| 1e-2 | 1e-5 | 42.7 |
| 1e-2 | 0 | 45.3 |

可以看到，当lr为0.1，weight_decay为0时，模型的准确率最高。
### step 5: 根据选定的参数，训练20个epoch，画出Loss曲线和Acc曲线。
![image](https://user-images.githubusercontent.com/42258146/141669914-0930bde9-e81c-479f-aeb5-1d67d9659eb7.png)
![image](https://user-images.githubusercontent.com/42258146/141669919-e641bf9c-f713-4cbf-80e3-938ceef462fe.png)
![image](https://user-images.githubusercontent.com/42258146/141669925-030d8a6f-e5d7-405a-b132-a5fd8d313a64.png)
![image](https://user-images.githubusercontent.com/42258146/141669931-0617b973-4d19-41f8-afd3-07c1c58f2510.png)

可见，训练集的准确率曲线与验证集的准确率曲线较为拟合。
