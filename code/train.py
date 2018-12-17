from net import *
import sys
sys.path.insert(0,'..')
from mxnet import autograd, gluon
from mxnet.gluon import loss as gloss
import time
import mxnet as mx
import mxnet.ndarray as nd
from collections import namedtuple

batch_size = 128
train_iter = mx.io.ImageRecordIter(path_imgrec='/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/tools/train.rec',
                                   path_imglist='/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/tools/train.lst',
                                   path_imgidx='/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/tools/train.idx',
                                   batch_size=batch_size,
                                   label_width = 30,
                                   resize=64,
                                   data_shape=(3,64,64),
                                   shuffle=True)
                                   
test_iter = mx.io.ImageRecordIter(path_imgrec='/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/tools/test.rec',
                                   path_imglist='/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/tools/test.lst',
                                   path_imgidx='/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/tools/test.idx',
                                   batch_size=batch_size,
                                   label_width = 30,
                                   resize=64,
                                   data_shape=(3,64,64),
                                   shuffle=False)

reg_loss = gloss.L2Loss()

def train(net, train_iter, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
    pre_time=time.time()

    for epoch in range(num_epochs):
        # reset the iterator each epoch
        train_iter.reset()
        
        # dynamic lr setting
        if epoch>0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)

        for i, batch in enumerate(train_iter):
            with autograd.record():
                output = net(batch.data[0].as_in_context(ctx))
                label = batch.label[0].as_in_context(ctx)
                loss = 100 * reg_loss(output,label)
            
            loss.backward()
            trainer.step(batch_size)
            train_loss = loss.mean().asscalar()
            
            cur_time = time.time()
            time_total = cur_time - pre_time
            log_info = ('epoch: %d,batch: %d, loss: %.4f, time: %.4f'
                     %(epoch, i ,train_loss,time_total))
            if i % 100 == 0:
                print(log_info)
        
        # export model
        if epoch % 10 == 0:
            net.export('/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/code/model_rgb/model_' + str(epoch))
            
            json_path='/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/code/model_rgb/model_' + str(epoch) + '-symbol.json'
            params_path='/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/code/model_rgb/model_' + str(epoch) + '-0000.params'

            symnet=mx.symbol.load(json_path)
            mod = mx.mod.Module(symbol=symnet,context=ctx)
            mod.bind(data_shapes=[('data',(batch_size,3,64,64))])
            mod.load_params(params_path)
            Batch=namedtuple('Batch',['data'])
            
            test_loss = 0
            test_iter.reset()
            for i, batch in enumerate(test_iter):
              data = batch.data[0].as_in_context(ctx)
              mod.forward(Batch([data]),is_train=False)
              pred = mod.get_outputs()
              label = batch.label[0].as_in_context(ctx)
              loss = 100 * reg_loss(output,label)
              test_loss += loss
              
            print('Mean Average Losses of testing :  %.4f'
                  % (test_loss.mean().asscalar() / i))

ctx = mx.gpu(1)
lr, wd = 0.001, 5e-4
num_epochs = 300
lr_period, lr_decay = int(num_epochs/5), 0.5
net = get_net(ctx, 30)
net.hybridize()
train(net,train_iter,num_epochs, lr,wd,ctx, lr_period,lr_decay)
