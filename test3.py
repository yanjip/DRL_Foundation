# class B():
#     def __init__(self,a1):
#         x=100
#         self.x2=x+1
#         a=a1.append(8)
#     def act(self):
#         x1=self.x+1
# x=B(a1=[2,])
# # x.a.append(8)
# # y=B()
# # print(x.a) #[2, 8]
# # print(y.a)  #[2, 8]
# print(x.x2)

import torch
# a=torch.randn([3,1])
# print(a)
# print(a[2].detach().numpy())



import random
# a=random.sample([3,4,5],3)  #均匀抽样
# print(a)

def x(a,b):
    print('hwll')
def r():
    return 1,2
b=r()
x(*r())