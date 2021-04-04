# Defining Network class and helper functions to run
import numpy as np
import tensorflow as tf

class Network:
  def __init__(self, arch, in_sh, lr, minimizer, init, mask):
    self.NCL = arch[-1][1][1]
    self.arch = arch
    self.lr = lr
    self.eps = 1e-10
    
    # Setup input and lists
    self.x = tf.placeholder(tf.float32, [None]+in_sh)
    self.y_ = tf.placeholder(tf.float32, [None])
    self.trv = tf.placeholder(dtype=bool, shape=())
    self.in_mv = [tf.placeholder(tf.float32, [2, None]) for _ in range(len(arch))]
    self.batchs = tf.shape(self.y_)[0]
    self.yint = tf.cast(self.y_, tf.int32)
    self.y_1h = tf.one_hot(self.yint, self.NCL)
    nclass = tf.reduce_sum(self.y_1h, axis=[0])
    self.avgmask = tf.reshape(self.y_1h/tf.tile(tf.reshape(nclass,[1, self.NCL]), [self.batchs, 1]), [self.batchs, 1, self.NCL])
    self.h = [self.x]
    self.wl = []
    self.ml = []
    self.mv = []
    
    # Parse arch and apply layers
    for l, (typ, par, _) in enumerate(self.arch):
      if typ=='f' or typ=='x':
        W = tf.Variable(tf.random.normal(par, stddev=np.sqrt(2./(par[0]+par[1]))))
        b = tf.Variable(tf.zeros([par[1]]))
        gamma = tf.Variable(tf.ones([par[0]]))
        beta = tf.Variable(tf.zeros([par[0]]))
        self.wl.append([W, b, gamma, beta])
        if mask:
          # Masks are made into variables for assignment, but not trained
          mW = tf.Variable(tf.ones(par), trainable=False)
          mb = tf.Variable(tf.ones([par[1]]), trainable=False)
          self.ml.append([mW, mb])
        in_reshape = tf.reshape(self.h[l],[self.batchs,par[0]])
        # Using param (trv=1 train, trv=0 val) to switch batch avg or stored avg
        mean, var = tf.nn.moments(in_reshape, axes=[0], keep_dims=True)
        self.mv.append(tf.concat([mean, var],0))
        in_BN = tf.cond(self.trv, lambda:(in_reshape-mean)/tf.sqrt(var + self.eps), lambda:(in_reshape-self.in_mv[l][0])/tf.sqrt(self.in_mv[l][1] + self.eps))
        in_BN = gamma*in_BN + beta
        if mask:
          tmph = tf.matmul(in_BN, W*mW)  + b*mb
        else:
          tmph = (tf.matmul(in_BN, W)  + b)
        if typ=='f':
          tmph = tf.nn.relu(tmph)
          self.h.append(tmph)
        else:
          self.y = tmph
          self.h.append(self.y)
          y_exp = tf.exp(self.y-tf.reduce_max(self.y,axis=[1], keepdims=True))
          y_sm = y_exp/tf.tile(tf.reduce_sum(y_exp+1e-10,[1], keepdims=True), [1,self.NCL])
          self.cost = tf.reduce_mean(-tf.reduce_sum(self.y_1h*tf.log(y_sm+1e-30),[1]))
          self.yclass = tf.argmax(self.y, axis=1, output_type=tf.int32)
      elif typ=='c':
        M = tf.Variable(tf.random.normal(par[:4], stddev=np.sqrt(2./(par[0]*par[1]*par[2]+par[3]))))
        self.wl.append(M)
        if mask:
          mM = tf.Variable(tf.ones(par[:4]))
          self.ml.append([mM])
        if mask:
          tmph = tf.nn.conv2d(self.h[l], M*mM[l], strides=par[4], padding='VALID', data_format='NCHW')
        else:
          tmph = (tf.nn.conv2d(self.h[l], M, strides=par[4], padding='VALID', data_format='NCHW'))
        tmph = tf.nn.relu(tmph)
        self.h.append(tmph)
      elif typ=='p':
        self.wl.append([])
        if mask:
          self.ml.append([])
        tmph = tf.nn.max_pool(self.h[l], ksize=par[0], strides=par[1], padding='VALID', data_format='NCHW')
        self.h.append(tmph)

    # Minimization and error
    self.global_step = tf.Variable(0, trainable=False)
    if minimizer == 'Adam':
      self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cost, global_step=self.global_step)
    elif minimizer == 'SGD':
      self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost, global_step=self.global_step)
    self.error = 1.0-tf.reduce_mean(tf.cast(tf.equal(self.yint, self.yclass), tf.float32))
    
  def load(self, sess, conf, mask):
    for l, (typ, par, _) in enumerate(self.arch):
      if typ=='f' or typ=='x':
        if conf:
          for w, refw in zip(self.wl[l],conf[l]):
            w.load(refw, sess)
        if mask:
          for m, refm in zip(self.ml[l],mask[l]):
            m.load(refm, sess)
      elif typ=='c':
        if conf:
          wl.load(conf[l], sess)
        if mask:
          ml.load(mask[l], sess)

  def run(self, sess, tr_x, tr_y, ts_x, ts_y, amv, data=False, savac=False, train=True, val=True):
    mvdict = {self.in_mv[l]: amv[l] for l in range(len(amv))}
    if data:
      if savac:        
        a, wei, tse = sess.run([self.h, self.wl, self.error], feed_dict={self.x: ts_x, self.y_: ts_y, self.trv: False, **mvdict})
      else:
        wei, tse = sess.run([self.wl, self.error], feed_dict={self.x: ts_x, self.y_: ts_y, self.trv: False, **mvdict})
    elif val:
      tse = sess.run(self.error, feed_dict={self.x: ts_x, self.y_: ts_y, self.trv: False, **mvdict})
    if train:
      tre, _, mv = sess.run([self.error, self.train_step, self.mv], feed_dict={self.x: tr_x, self.y_: tr_y, self.trv: True, **mvdict})
    else:
      tre = 0.0
    if data:
      if savac:
        return a, aa, wei, tse, tre, mv
      else:
        return wei, tse, tre, mv
    elif val:
      return tse, tre, mv
    else:
      return tre, mv
