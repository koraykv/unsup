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
   type = "command",
   build_command = [[
   		 cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}