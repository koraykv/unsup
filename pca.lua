
-- PCA using covariance matrix
-- x is supposed to be MxN matrix, where M samples(trials) and each sample(trial) is N dim
-- returns the 
function unsup.pcacov(x)
   local mean = torch.mean(x,1)
   local xm = x - torch.ger(torch.ones(x:size(1)),mean:squeeze())
   local c = torch.mm(xm:t(),xm)
   c:div(x:size(1)-1)
   local ce,cv = torch.symeig(c,'V')
   return ce,cv
end
