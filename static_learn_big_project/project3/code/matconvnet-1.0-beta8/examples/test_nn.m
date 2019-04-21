%
%
function res=test_nn(images,batch,net)

im=getBatch(images,batch);
res=vl_simplenn(net,im);


function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.data(:,:,:,batch) ;
labels = imdb.labels(1,batch) ;