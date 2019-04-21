%
%
clear all;
load('../../../data/imretreival/imdb.mat');
run('../matlab/vl_setupnn.m');
load('../../../data/imretreival/net-epoch-20.mat');

batchSize=50;
for t=1:41
    batch=((t-1)*batchSize+1):(t*batchSize);
    temp_res=test_nn(images,batch,net);
    y=vl_nnsoftmax(temp_res(12).x);
    [~,class]=max(y);
    temp_res(end+1).x=reshape(class,[batchSize,1]);
    result{t}=temp_res;
    clear temp_res;
end

save('./result.mat','result');