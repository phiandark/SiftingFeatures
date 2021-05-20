# Defining helper functions to handle masks and parameters
import numpy as np

def archparse(archstr, prstr, tdata):
  lays = archstr.split(":")
  if len(prstr)==0:
    # If not given all True
    pr = [True for _ in lays]
  else:
    # Parse string
    pr = [True if t=='T' else False for t in prstr.split(":")]
    # Pad with True if short
    if len(pr)<len(lays):
      pr.extend([False for _ in range(len(lays)-len(pr))])
  cin = tdata.c
  xs = tdata.w
  ys = tdata.h
  insize = xs*ys*cin
  # Build architecture
  arch = []
  for l, p in zip(lays,pr):
    if l[0]=='f':
      outsize = int(l[1:])
      arch.append(['f', [insize,outsize], p])
      insize = outsize
    elif l[0]=='c':
      pars = [int(n) for n in l[1:].split(",")]
      k = pars[0]
      cout = pars[1]
      if len(pars)>2:
        s = pars[2]
      else:
        s = 1
      arch.append(['c', [k,k,cin,cout,s], p])
      cin = cout
      xs = int(np.ceil((xs-k+1)/s))
      ys = int(np.ceil((ys-k+1)/s))
      insize = xs*ys*cin
    elif l[0]=='p':
      k, s = [int(n) for n in l[1:].split(",")]
      arch.append(['p',[k,s], False])
      xs = int(np.ceil((xs-k+1)/s))
      ys = int(np.ceil((ys-k+1)/s))
      insize = xs*ys*cin
    elif l[0]=='x':
      arch.append(['x',[insize,tdata.ncl], p])
    else:
      print("Unknown layer type: "+l)
      exit()
  return arch

def get_mask(ic, arch, ratios, maskold=None):
  maskl = []
  ri = 0
  for l, (typ, par, prun) in enumerate(arch):
    if typ=='f' or typ=='x':
      if prun:
        tw = np.abs(ic[l][0])
        tb = np.abs(ic[l][1])
        if maskold:
          tw *= maskold[l][0]
          tb *= maskold[l][1]
        wlist = np.concatenate((tw.flatten(),tb))
        ncutoff = int(ratios[ri]*len(wlist))
        ri += 1
        wlist.sort()
        thrs = wlist[ncutoff]
        wmask = np.where(tw<thrs, 0.0, 1.0).astype(np.float32)
        bmask = np.where(tb<thrs, 0.0, 1.0).astype(np.float32)
        maskl.append([wmask, bmask])
      elif maskold:
        maskl.append(maskold[l])
      else: 
        maskl.append([np.ones(np.shape(ic[l][0])),np.ones(np.shape(ic[l][1]))])
    elif typ=='c':
      if prun:
        tm = np.abs(ic[l][0])
        if maskold:
          tm *= maskold[l][0]
        wlist = tm.flatten()
        ncutoff = int(ratios[ri]*len(wlist))
        ri += 1
        wlist.sort()
        thrs = wlist[ncutoff]
        maskl.append([np.where(tm<thrs, 0.0, 1.0).astype(np.float32)])
      else: 
        maskl.append([np.ones(np.shape(ic[l]))])
    elif typ=='p':
      maskl.append([])
  return maskl


def make_mask(arch, spars):
  imask = []
  si = 0
  for l, (typ, par, prun) in enumerate(arch):
    if typ=='f' or typ=='x':
      if prun:
        wsp = np.where(np.random.uniform(0.0,1.0,(par[0],par[1]))<spars[si],0,1).astype(np.float32)
        imask.append([wsp, np.ones((par[1]),dtype=np.float32)])
        si += 1
      else:
        imask.append([np.ones((par[0],par[1]),dtype=np.float32), np.ones((par[1]),dtype=np.float32)])
    elif typ=='c':
      if prun:
        imask.append([np.where(np.random.uniform(0.0,1.0,(par[0],par[1],par[2],par[3]))<spars[si],0,1).astype(np.float32)])
        si += 1
      else:
        imask.append([np.ones((par[0],par[1],par[2],par[3]),dtype=np.float32)])
    elif typ=='p':
      imask.append([])
    else:
      print("Layer {} not supported".format(typ))
  return imask
