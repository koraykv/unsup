UNSUP
=====

A package for unsupervised learning in Torch.

Provides modules that are compatible with `nn` (`LinearPsd`, `ConvPsd`, `AutoEncoder`, ...),
and self-contained algorithms (`k-means`, `PCA`).

Requirements
------------

Basic dependencies:

  * Torch7 (github.com/andresy/torch)
  * kex    (github.com/koraykv/tools)
  * optim  (github.cim/koraykv/optim)

To run the demo scripts, you also need the following:

  * image (github.com/clementfarabet/lua---image)
  * sys   (github.com/clementfarabet/lua---sys)
  * xlua  (github.com/clementfarabet/lua---xlua)

Installation
------------

Build/Install:

  * Install Torch7 (refer to its own documentation).
  * clone all other repos (including this one) into dev directory of Torch7.
  * Rebuild torch, it will include all these projects too.

Alternatively, you can use torch's package manager. Once
Torch is installed, you can install `unsup`: `$ torch-pkg install unsup`.
