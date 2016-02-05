
-- PCA using covariance matrix
-- x is supposed to be MxN matrix, where M is the number of samples (trials) and each sample (trial) is N dim
-- returns the eigen values and vectors of the covariance matrix in *INCREASING* order
function unsup.pcacov(x)
   local mean = torch.mean(x,1)
   local xm = x - mean:expandAs(x)
   local c = torch.mm(xm:t(),xm)
   c:div(x:size(1)-1)
   local ce,cv = torch.symeig(c,'V')
   return ce,cv
end

-- PCA using SVD
-- x is supposed to be MxN matrix, where M is the number of samples (trials) and each sample (trial) is N dim
-- returns the eigen values and vectors of the covariance matrix in decreasing order
function unsup.pca(x)
   local mean = torch.mean(x,1)
   local xm = x - mean:expandAs(x)
   xm:div(math.sqrt(x:size(1)-1))
   local w,s,v = torch.svd(xm:t())
   s:cmul(s)
   return s,w
end
