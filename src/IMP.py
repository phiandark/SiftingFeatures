import numpy as np
import tensorflow as tf
import pickle
import argparse
import configparser
import os
import multiprocessing
import time
import copy
import sys

from Network import *
from Data import *
from Utils import *

# To allow GPU memory management
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# To select a specific GPU device
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
  # Parsing command line (name of input file)
  parser = argparse.ArgumentParser()
  parser.add_argument("input", type=str, help="Input file")
  args = parser.parse_args()
  #Reading input file sections
  print('Reading input {}'.format(args.input))
  config = configparser.ConfigParser()
  config.read(args.input)

  # Input/output
  c_io = config['IO']
  save_dir = c_io.get('save_dir', '.')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  prefix = c_io.get('prefix', 'IMP')
  save_prefix = save_dir+"/"+prefix
  logfile = save_prefix+"_out.dat"
  of = open(logfile, 'w')

  # Dataset
  c_data = config['Data']
  dataset = c_data.get('dataset', 'ImageNet32')
  dataloc = c_data.get('dataloc', './ImageNet')
  preload = c_data.getboolean('preload', True)
  data_ratio = np.clip(c_data.getfloat('data_ratio', 1.0), 0.0, 1.0)
  reclass = c_data.get('reclass', 'n')

  # Network
  c_net = config['Network']
  archstr = c_net.get('arch', '')
  steps = c_net.getint('steps', 1e5)
  batch_size = c_net.getint('batch_size', 1000)
  eval_size = c_net.getint('val_size', 1000)
  minzer = c_net.get('minimizer', 'SGD')
  lr = c_net.getfloat('learning_rate', 0.1)
  ratio = c_net.getfloat('init_sparsity', 0.0)
  val_each = c_net.getint('validate_step', 100)
  bnbeta = c_net.getfloat('batch_norm_beta' , 0.99)
  # Data to restart from previous file
  restart = c_net.getboolean('restart', False)
  if restart:
    itst = c_net.getint('rest_it', 1)
    rest_file_wei = c_net.get('rest_weights_file', save_dir+"/IMP_itdata_0.pkl")
    rest_file_mask = c_net.get('rest_mask_file', save_dir+"/IMP_itdata_"+str(itst-1)+".pkl")
    rest_file_step = c_net.getint('restart_weights_step', -1)

  # IMP parameters
  c_IMP = config['IMP']
  prstr = c_IMP.get('prune', '')
  rat_factor = c_IMP.getfloat('prune_ratio', 0.3)
  st_rest = c_IMP.getint('weights_step', 1000)
  st_wei = c_IMP.getint('prune_step', steps)
  nnode_stop = c_IMP.getfloat('stop_ratio', 0.0)
  itmax = c_IMP.getint('max_iterations', 1000)
  # Threshold to consider a node disconnected
  NUconn = 0

  np.random.seed(int(time.time()))
  
  print('Input read. Loading data...')
  data = Data(dataset, dataloc, eval_size, ratio=data_ratio, preload=preload, reclass=reclass)
  print('Data read. Building network...')
  # Setting up network
  arch = archparse(archstr, prstr, data)
  prunable = [l-1 for l, lay in enumerate(arch) if lay[2]] #List of prunable
  lprun = len(prunable)
  ratios = ratio*np.ones(lprun)

  if restart:
    # Load weights and masks if this is a restart
    with open(rest_file_wei, 'rb') as f:
      initdata = pickle.load(f)
    initconf = initdata[7][0]
    with open(rest_file_mask, 'rb') as f:
      [_, ratios, Tpoints,_,_,_,_, weil ,_, ml] = pickle.load(f)
    finalconf = weil[rest_file_step]
    ratios += rat_factor*(1-ratios)
    mask = get_mask(finalconf, arch, ratios, ml)
    it = itst
    if st_wei<steps:
      Tpoints = [st_wei, steps]
    else:
      Tpoints = [steps]
    initmv = None
  else:
    # Create possibly sparse mask, init all
    mask = make_mask(arch, ratios)
    initconf = None
    initmv = None
    if st_wei<steps:
      Tpoints = [st_rest, st_wei, steps]
    else:
      Tpoints = [st_rest, steps]
    it = 0

  in_sh = [data.c, data.h, data.w]
  net = Network(arch, in_sh, lr, minzer, initconf, mask)
  NNConn = np.min(np.asarray([len(np.where(np.sum(mask[l][0],axis=0)>NUconn)[0])/np.shape(mask[l][0])[1] for l in prunable]))
  ratl = []
  mins = []
  mll = []
  
  print('Starting training.')
  # Running everything in the same session, or memory is not freed
  with tf.Session() as sess:
    while NNConn>nnode_stop and it<itmax:
      sl = []
      tl = []
      sel = []
      el = []
      weil = []
      actl = []
      amvl = []
      tf.global_variables_initializer().run()
      net.load(sess, initconf, mask)
      if initmv:
        amv = initmv
      else:
        amv = []
        for l, (typ, par, _) in enumerate(arch):
          if typ=='f' or typ=='x':
            amv.append(np.concatenate((np.zeros((1,par[0]),dtype=np.float32),np.ones((1,par[0]),dtype=np.float32)),axis=0))

      for st in range(steps+1):
        batch_xs, batch_ys = data.getbatch(batch_size)
        if st in set(Tpoints):
          ww, ee, te, mv = net.run(sess, batch_xs, batch_ys, data.e_in, data.e_lab, amv, True)
          weil.append(ww)
          amvl.append(amv)
        elif st%val_each==0:
          ee, te, mv = net.run(sess, batch_xs, batch_ys, data.e_in, data.e_lab, amv, False)
        else:
          te, mv = net.run(sess, batch_xs, batch_ys, data.e_in, data.e_lab, amv, False, val=False)
        sl.append(st)
        tl.append(te)
        if st%val_each==0:
          print('Step: {:7d}, TE: {:5.3f}, VE: {:5.3f}'.format(st, te, ee), file=of, end="\r", flush=True)
          el.append(ee)
          sel.append(st)
        if st == 0:
          amv = mv.copy()
        else:
          # Running average
          for l, tmv in enumerate(mv):
            amv[l] = bnbeta*amv[l] + (1-bnbeta)*tmv

      emin = np.min(el)
      print("Ratios: {}; more than {} connections: {}; Emin: {}".format(ratios, NUconn, NNConn, emin), file=of)
      print("Iteration {} -- Ratio: {}; Emin: {}".format(it, np.mean(ratios), emin))
      ratl.append(np.copy(ratios))
      mins.append(emin)
      data2save = [arch, ratios, Tpoints, sl, sel, tl, el, weil, amvl, mask]
      with open(save_prefix+"_itdata_"+str(it)+".pkl", 'wb') as f:
        pickle.dump(data2save, f)
      
      if it==0:
        initconf = weil[0]
        Tpoints.pop(0)

      ratios += rat_factor*(1-ratios)
      if st_wei<steps:
        finalconf = weil[-2]
      else:
        finalconf = weil[-1]
      mask = get_mask(finalconf, arch, ratios, mask)
      NNConn = np.min(np.asarray([len(np.where(np.sum(mask[l][0],axis=0)>NUconn)[0])/np.shape(mask[l][0])[1] for l in prunable]))
      it += 1

  data2save = [arch, ratl, mins]
  with open(save_prefix+"_findata.pkl", 'wb') as f:
    pickle.dump(data2save, f)
  of.close()

if __name__ == '__main__':
  main()
