require 'torch'
require 'unsup'
require 'image'


torch.manualSeed(0)

mytest = {}


local function pearson_correlation_coefficient(data)
    local corr = torch.Tensor(data:size(2), data:size(2))

    local means = torch.mean(data, 1):squeeze()
    local stds = torch.std(data, 1):squeeze():add(1e-10)

    for i=1,data:size(2) do
        for j=i,data:size(2) do
            corr[j][i] = ((data[{{}, i}] - means[i]) * (data[{{}, j}] - means[j])) / stds[i] / stds[j] / data:size(1)
            corr[i][j] = corr[j][i]
        end
    end
    --print(torch.min(corr), torch.mean(corr), torch.max(corr))
    return corr
end
 

function mytest.zca_whiten()
    local gaussian_white_data = torch.randn(1000, 300)
    --[[ 
    do
        image.display({image=gaussian_white_data, zoom=2, legend='gaussian_white_data'})
        image.display({image=pearson_correlation_coefficient(gaussian_white_data), zoom=2, min=-1, max=1,   legend='corr_gaussian_white_data'})
        gnuplot.figure()
        gnuplot.hist(pearson_correlation_coefficient(gaussian_white_data), 100)
    end
    --]]

    local l = nn.Linear(gaussian_white_data:size(2),gaussian_white_data:size(2))
    l.weight:copy(torch.lt(torch.rand(l.weight:size()), 1/math.sqrt(gaussian_white_data:size(2))):double())
    l.bias:fill(0)
    local linearly_correlated_data = l:forward(gaussian_white_data)
    --[[ 
    do 
        image.display(l.weight)
        image.display({image=linearly_correlated_data, zoom=2, legend='linearly_correlated_data'})
        image.display({image=pearson_correlation_coefficient(linearly_correlated_data), min=-1, max=1, zoom=2, legend='corr_linearly_correlated_data'})
        gnuplot.figure()
        gnuplot.hist(pearson_correlation_coefficient(linearly_correlated_data), 100)
    end
    --]]

    local zca_whitened_data
    zca_whitened_data, means, P, invP  = unsup.zca_whiten(linearly_correlated_data)
    --[[ 
    do
        image.display({image=zca_whitened_data, zoom=2, legend='zca_whitened_data'})
        image.display({image=pearson_correlation_coefficient(zca_whitened_data), zoom=2, min=-1, max=1, legend='corr_zca_whitened_data'})
        gnuplot.figure()
        gnuplot.hist(pearson_correlation_coefficient(zca_whitened_data), 100)
    end 
    --]]
    local stat = torch.mean(torch.pow(pearson_correlation_coefficient(zca_whitened_data)
                                            - pearson_correlation_coefficient(gaussian_white_data), 2))
    tester:assertlt(stat, 1e-3, 'corr_diff < 1e-3')
end




tester = torch.Tester()
tester:add(mytest)
tester:run()


