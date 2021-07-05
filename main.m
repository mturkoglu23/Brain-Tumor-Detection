clear;clc;

layer1 ='fc1000';
layer2 ='fc6';
net1 = densenet201;
net2 =alexnet;

%%
imds = imageDatastore('E:\10_07_2020\brain mr\yeni data\',...
    'IncludeSubfolders',true,...
    'LabelSource','FolderNames');
imds.ReadFcn = @(loc)imresize(imread(loc),[300 300]);
[imdsTrain,idmsTest] = splitEachLabel(imds,0.8,'randomized');

uzunluk=numel(imds.Labels);

for i=1:uzunluk
    i
 img1=readimage(imds,i);
aa=size(img1);
 if length(aa)==2
   img1=cat(3,img1,img1,img1);
 end
 img1=imresize(img1,[224 224]);
img2=imresize(img1,[227 227]);

    dense_Feats_test(:,i) = activations(net1,img1,layer1);
    alex_Feats_test(:,i) = activations(net2,img2,layer2);

end
labels=imds.Labels;

Y1=double(labels);
X1=[dense_Feats_test',alex_Feats_test'];

[rr1,ww1]=fscmrmr(X1,Y1); // mRmR feature selection 

for ii=1:2000
    new_feat(:,ii)=X1(:,rr1(ii));
end

feat=[new_feat,Y1]; // The MATLAB Classification Learner toolbox is used for the application of Bayesian and SVM methods.


