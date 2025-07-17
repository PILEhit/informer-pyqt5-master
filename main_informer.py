import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
parser.add_argument('--model', type=str, required=False, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
parser.add_argument('--data', type=str, required=False, default='beartem0529', help='data')
# 任务指定
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='data_align_tpt_0207.csv', help='data file')    
parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
# 任务模式指定：多对多、单对单、多对单
parser.add_argument('--target', type=str, default='动量轮FMW2轴承温度', help='target feature in S or MS task')
# 目标指定，指定某一列名
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
# 频率指定
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
# 编码器输入长度96
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
# 解码器从编码器输入的后半段开始截取，截取48个
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
# 剩下的24是要预测的
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
parser.add_argument('--enc_in', type=int, default=33, help='encoder input size')
# 编码器输入维度，7列数据
parser.add_argument('--dec_in', type=int, default=33, help='decoder input size')
# 解码器输入维度，7列数据
parser.add_argument('--c_out', type=int, default=33, help='output size')
# 输出维度也是7
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
# 在模型中，向量维度开始为512
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
# 注意力头，即有多少个注意力角度
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
# 编码器multihead层数
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
# 解码器multihead层数
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
# 层个数，321是指标号，这三个encoder层，每层都要走2个多头注意力机制
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
# 是否使用下采样（这里叫蒸馏）
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
# 时间特征编码方式
# 提取时序数据特征，将一个时间点的多维数据扩充至更高维，使用1x1卷积
parser.add_argument('--activation', type=str, default='gelu',help='activation')
# 激活函数使用gelu
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', default=False, help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
# 指定输入特征
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers') # windows默认为0
parser.add_argument('--itr', type=int, default=1, help='experiments times')
# 试验次数，做两次实验test0、test1，总训练次数是2*epoch，因为要做对比试验，比较2次训练结果
parser.add_argument('--train_epochs', type=int, default=24, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
# 32组数据0-95~~~31-126
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
# 若loss不收敛则停止训练
parser.add_argument('--learning_rate', type=float, default=0.00001,help='optimizer learning rate')
parser.add_argument('--des', type=str, default='train',help='exp description: train,test,pred')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
# 损失函数采用mse 均方误差
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
# 启用混合精度训练
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
# 输出反归一化
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# 可以执行数据预测的菜单
data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
    'TPT':{'data':'太敏01.csv','T':'卫星位置Z方向','M':[4,4,4],'S':[1,1,1],'MS':[4,4,2]},
    'beartem':{'data':'data_align_tpt_0208.csv','T':'动量轮FMW2轴承温度','M':[9,9,9],'S':[1,1,1],'MS':[9,9,1]},
    'beartem1':{'data':'data_align_tpt_0206.csv','T':'动量轮FMW2轴承温度','M':[33,33,33],'S':[1,1,1],'MS':[33,33,1]},
    'beartem0529':{'data':'data_tpt_0206gbk_final_toplist.csv','T':'动量轮FMW2轴承温度','M':[11,11,11],'S':[1,1,1],'MS':[11,11,1]}
}

if args.data in data_parser.keys():
    # 如果args指定的data存在于字典中
    data_info = data_parser[args.data]
    # 取出那个data对象,不输入则为默认数据
    args.data_path = data_info['data']
    # 取出路径与文件名
    args.target = data_info['T']
    # 目标是预测OT这一列（默认为OT），这是单单预测或者多单预测中可以指定的列，具体是什么需要打开csv文件查看
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]
    # features代表执行的三种不同的预测任务，informer的输入维度即为编码器enc输入维度
    # informer输出的预测对象的维度就是译码器dec的输入维度，c_out代表网络整体输出

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
# 编码器结构，默认结构为1,2,3共三层使用者是否使用空格的输入习惯
args.detail_freq = args.freq
# 频率：默认按照小时进行时间特征编码，具体频率可输入2h
args.freq = args.freq[-1:]
# 如果输入了2h，最终也只读取末尾这个字母

print('Args in experiment:')
print(args)

Exp = Exp_Informer
# 加载模型

# 记录实验
for ii in range(args.itr):
    # setting record of experiments
    # 记录itr=2次实验：test0、test1
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, ii)
    # 记录它的各种信息，保存文件夹名

    exp = Exp(args) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
preds = np.load('./results/'+setting+'/pred.npy')
trues = np.load('./results/'+setting+'/true.npy')

plt.figure()
plt.plot(trues[:,0,-1], label='GroundTruth')
plt.plot(preds[:,0,-1], label='Prediction')
plt.legend()
plt.show()

pred_output = pd.DataFrame(np.array(preds[:,0,-1]),columns=['pred_output'])
pred_output['true_output'] = np.array(trues[:,0,-1])
pred_output.tail()

plt.figure(figsize=(20,5))
plt.plot(pred_output.index,pred_output[['pred_output','true_output']])
plt.legend(['pred_output','true_output'])
# %%
