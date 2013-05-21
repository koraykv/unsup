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
    return corr
end
 

function mytest.zca_whiten()
    local gaussian_white_data = torch.randn(1000, 300)
    
    local l = nn.Linear(gaussian_white_data:size(2),gaussian_white_data:size(2))
    l.weight:copy(torch.lt(torch.rand(l.weight:size()), 1/math.sqrt(gaussian_white_data:size(2))):double())
    l.bias:fill(0)
    local linearly_correlated_data = l:forward(gaussian_white_data)
    
    local zca_whitened_data
    zca_whitened_data, means, P, invP  = unsup.zca_whiten(linearly_correlated_data)
    local stat = torch.mean(torch.pow(pearson_correlation_coefficient(zca_whitened_data)
                                            - pearson_correlation_coefficient(gaussian_white_data), 2))
    tester:assertlt(stat, 1e-2, 'corr_diff < 1e-2')
end


function mytest.zca_colour()
    local gaussian_white_data = torch.randn(1000, 300)
    
    local l = nn.Linear(gaussian_white_data:size(2),gaussian_white_data:size(2))
    l.weight:copy(torch.lt(torch.rand(l.weight:size()), 1/math.sqrt(gaussian_white_data:size(2))):double())
    l.bias:fill(0)
    local linearly_correlated_data = l:forward(gaussian_white_data)
    
    local zca_whitened_data
    zca_whitened_data, means, P, invP  = unsup.zca_whiten(linearly_correlated_data)
   
    -- colour data
    local coloured_data
    coloured_data = unsup.zca_colour(zca_whitened_data, means, P, invP)
    local stat = torch.max(torch.abs(linearly_correlated_data - coloured_data))
    tester:assertlt(stat, 1e-10, 'rec_diff < 1e-10')
end


function mytest.zca_layer()
    local gaussian_white_data = torch.randn(1000, 300)
    local l = nn.Linear(gaussian_white_data:size(2),gaussian_white_data:size(2))
    l.weight:copy(torch.lt(torch.rand(l.weight:size()), 1/math.sqrt(gaussian_white_data:size(2))):double())
    l.bias:fill(0)
    local linearly_correlated_data = l:forward(gaussian_white_data)
    local zca_whitened_data
    zca_whitened_data, means, P, invP  = unsup.zca_whiten(linearly_correlated_data)
    local layer = unsup.zca_layer(linearly_correlated_data)
    local layer_output = layer:forward(linearly_correlated_data)
    local stat = torch.max(torch.abs(zca_whitened_data - layer_output))
    tester:assertlt(stat, 1e-10, 'rec_diff < 1e-10')
end


function mytest.pca_whiten()
    local gaussian_white_data = torch.randn(1000, 300)
    
    local l = nn.Linear(gaussian_white_data:size(2),gaussian_white_data:size(2))
    l.weight:copy(torch.lt(torch.rand(l.weight:size()), 1/math.sqrt(gaussian_white_data:size(2))):double())
    l.bias:fill(0)
    local linearly_correlated_data = l:forward(gaussian_white_data)
    
    local pca_whitened_data
    pca_whitened_data, means, P, invP  = unsup.pca_whiten(linearly_correlated_data)
    local stat = torch.mean(torch.pow(pearson_correlation_coefficient(pca_whitened_data)
                                            - pearson_correlation_coefficient(gaussian_white_data), 2))
    tester:assertlt(stat, 1e-2, 'corr_diff < 1e-2')
end


function mytest.pca_colour()
    local gaussian_white_data = torch.randn(1000, 300)
    
    local l = nn.Linear(gaussian_white_data:size(2),gaussian_white_data:size(2))
    l.weight:copy(torch.lt(torch.rand(l.weight:size()), 1/math.sqrt(gaussian_white_data:size(2))):double())
    l.bias:fill(0)
    local linearly_correlated_data = l:forward(gaussian_white_data)
    
    local pca_whitened_data
    pca_whitened_data, means, P, invP  = unsup.pca_whiten(linearly_correlated_data)
   
    -- colour data
    local coloured_data
    coloured_data = unsup.pca_colour(pca_whitened_data, means, P, invP)
    local stat = torch.max(torch.abs(linearly_correlated_data - coloured_data))
    tester:assertlt(stat, 1e-10, 'rec_diff < 1e-10')
end


function mytest.pca_layer()
    local gaussian_white_data = torch.randn(1000, 300)
    local l = nn.Linear(gaussian_white_data:size(2),gaussian_white_data:size(2))
    l.weight:copy(torch.lt(torch.rand(l.weight:size()), 1/math.sqrt(gaussian_white_data:size(2))):double())
    l.bias:fill(0)
    local linearly_correlated_data = l:forward(gaussian_white_data)
    local pca_whitened_data
    pca_whitened_data, means, P, invP  = unsup.pca_whiten(linearly_correlated_data)
    local layer = unsup.pca_layer(linearly_correlated_data)
    local layer_output = layer:forward(linearly_correlated_data)
    local stat = torch.max(torch.abs(pca_whitened_data - layer_output))
    tester:assertlt(stat, 1e-10, 'rec_diff < 1e-10')
end


tester = torch.Tester()
tester:add(mytest)
tester:run()


