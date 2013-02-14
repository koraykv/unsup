package = "unsup"
version = "0.1-0"

source = {
   url = "git://github.com/koraykv/unsup",
   tag = "master"
}

description = {
   summary = "A package for unsupervised learning in Torch",
   detailed = [[
Provides modules that are compatible with nn (LinearPsd, ConvPsd, AutoEncoder, ...), and self-contained algorithms (k-means, PCA).
   ]],
   homepage = "https://github.com/koraykv/unsup",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "xlua >= 1.0",
   "optim >= 1.0"
}

build = {
   type = "cmake",
      variables = {
            LUAROCKS_PREFIX = "$(PREFIX)"
      }
}