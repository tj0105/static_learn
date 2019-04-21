%
%

clear all;
load('./result.mat');
load('../../../data/imretreival/imdb.mat');

labels=images.labels;
labels=labels';

for i=1:length(result)
    all_feature(:,(i-1)*50+1:i*50)=reshape(vl_nnsoftmax(result{1,i}(12).x),[41,50]);
end

index=find(labels~=14);
labels=labels(index);
all_feature=all_feature(:,index);


evaluation_10=double(zeros(2000,4));
evaluation_20=double(zeros(2000,4));
evaluation_50=double(zeros(2000,4));
evaluation_100=double(zeros(2000,4));

k=10.0;
tic
for i=1:2000
    temp_feature=all_feature;
    temp_feature=temp_feature-repmat(temp_feature(:,i),1,2000);
    temp_feature=temp_feature.*temp_feature;
    temp_feature=sum(temp_feature);
    temp_feature=temp_feature';
    temp_feature=[temp_feature (1:2000)'];
    temp_feature=sortrows(temp_feature,1);
    temp_label=labels(temp_feature(1:k,2));
    correct=length(find(temp_label==labels(i)));
    evaluation_10(i,1)=correct/double(k);
    evaluation_10(i,2)=correct/50.0;
    evaluation_10(i,3)=correct*2/double(k+50.0);
    evaluation_10(i,4)=0.2*sum(1./find(temp_label==labels(i)));
end
toc
    
k=20.0;
tic
for i=1:2000
    temp_feature=all_feature;
    temp_feature=temp_feature-repmat(temp_feature(:,i),1,2000);
    temp_feature=temp_feature.*temp_feature;
    temp_feature=sum(temp_feature);
    temp_feature=temp_feature';
    temp_feature=[temp_feature (1:2000)'];
    temp_feature=sortrows(temp_feature,1);
    temp_label=labels(temp_feature(1:k,2));
    correct=length(find(temp_label==labels(i)));
    evaluation_20(i,1)=correct/double(k);
    evaluation_20(i,2)=correct/50.0;
    evaluation_20(i,3)=correct*2/double(k+50.0);
    evaluation_20(i,4)=0.2*sum(1./find(temp_label==labels(i)));
end
toc

k=50.0;
tic
for i=1:2000
    temp_feature=all_feature;
    temp_feature=temp_feature-repmat(temp_feature(:,i),1,2000);
    temp_feature=temp_feature.*temp_feature;
    temp_feature=sum(temp_feature);
    temp_feature=temp_feature';
    temp_feature=[temp_feature (1:2000)'];
    temp_feature=sortrows(temp_feature,1);
    temp_label=labels(temp_feature(1:k,2));
    correct=length(find(temp_label==labels(i)));
    evaluation_50(i,1)=correct/double(k);
    evaluation_50(i,2)=correct/50.0;
    evaluation_50(i,3)=correct*2/double(k+50.0);
    evaluation_50(i,4)=0.2*sum(1./find(temp_label==labels(i)));
end
toc

k=100.0;
tic
for i=1:2000
    temp_feature=all_feature;
    temp_feature=temp_feature-repmat(temp_feature(:,i),1,2000);
    temp_feature=temp_feature.*temp_feature;
    temp_feature=sum(temp_feature);
    temp_feature=temp_feature';
    temp_feature=[temp_feature (1:2000)'];
    temp_feature=sortrows(temp_feature,1);
    temp_label=labels(temp_feature(1:k,2));
    correct=length(find(temp_label==labels(i)));
    evaluation_100(i,1)=correct/double(k);
    evaluation_100(i,2)=correct/50.0;
    evaluation_100(i,3)=correct*2/double(k+50.0);
    evaluation_100(i,4)=0.2*sum(1./find(temp_label==labels(i)));
end
toc


