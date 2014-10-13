require 'torch'
require 'dataset'
require 'dataset/cifar10'
require 'image'
require 'unsup'



ds = cifar10.raw_dataset().data:narrow(1,1,100):narrow(2,1,1):clone()
collectgarbage()
print({ds})
image.display{image=ds, zoom=2, nrow=10}


wds, m, P, invP = unsup.zca_whiten(ds)
print({wds})
image.display{image=wds, zoom=2, nrow=10}

ods = unsup.zca_colour(wds, m, P, invP)
print({ods})
image.display{image=ods, zoom=2, nrow=10}

