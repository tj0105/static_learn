%
%
function my_cnn(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR

% run(fullfile(fileparts(mfilename('fullpath')), ...
%   '..', 'matlab', 'vl_setupnn.m')) ;

run('./matconvnet-1.0-beta8/matlab/vl_setupnn.m');

opts.expDir = '../data/imretreival/';
opts.imdbPath = '../data/imretreival/imdb.mat';
opts.train.batchSize = 50 ;
opts.train.numEpochs = 20 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = [0.001*ones(1, 12) 0.0001*ones(1,6) 0.00001] ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

  imdb = load(opts.imdbPath) ;

% Define network CIFAR10-quick
net.layers = {} ;

% 1 conv1
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 1e-4*randn(5,5,3,32, 'single'), ...
  'biases', zeros(1, 32, 'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 2) ;

% 2 pool1 (max pool)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

% 3 relu1
net.layers{end+1} = struct('type', 'relu') ;

% 4 conv2
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 0.01*randn(5,5,32,32, 'single'),...
  'biases', zeros(1,32,'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 2) ;

% 5 relu2
net.layers{end+1} = struct('type', 'relu') ;

% 6 pool2 (avg pool)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ; % Emulate caffe

% 7 conv3
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 0.01*randn(5,5,32,64, 'single'),...
  'biases', zeros(1,64,'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 2) ;

% 8 relu3
net.layers{end+1} = struct('type', 'relu') ;

% 9 pool3 (avg pool)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ; % Emulate caffe

% 10 ip1
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 0.1*randn(4,4,64,64, 'single'),...
  'biases', zeros(1,64,'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 0) ;

% 11 ip2
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.1*randn(1,1,64,41, 'single'),...
                           'biases', zeros(1,41,'single'), ...
                           'filtersLearningRate', 1, ...
                           'biasesLearningRate', 2, ...
                           'stride', 1, ...
                           'pad', 0) ;
% 12 loss
net.layers{end+1} = struct('type', 'softmaxloss') ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

% Take the mean out and make GPU if needed
% imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4)) ;
% if opts.train.useGpu
%   imdb.images.data = gpuArray(imdb.images.data) ;
% end

[net,info] = cnn_train(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 1));

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

