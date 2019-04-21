%
% processes for data used in CNN
% by heathcliff

function imdb=getimdb(base_dir)

folder=dir(base_dir);
data_num=1;

for i=1:length(folder)-2
    classes{i}=folder(i+2).name;
    file_name=dir([base_dir classes{i} '/*.jpg']);
    for k=1:length(file_name)
        img=imread([base_dir classes{i} '/' file_name(k).name]);
        if(length(size(img))==2)
            temp_data(:,:,1)=img;
            temp_data(:,:,2)=img;
            temp_data(:,:,3)=img;
        else
            temp_data=img;
        end
        data(:,:,:,data_num)=single(imresize(temp_data,[32,32]));
        labels{data_num}=i;
        sets{data_num}=1;
        data_num=data_num+1;
        clear temp_data;
    end
end

dataMean = mean(data, 4);
data = bsxfun(@minus, data, dataMean);

index=randperm(length(labels));

imdb.images.data = data(:,:,:,index) ;
imdb.images.data_mean = dataMean;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.labels = imdb.images.labels(index);
imdb.images.set = cat(2, sets{:});
imdb.meta.sets = {'train', 'val', 'test'};
imdb.meta.classes=classes;
