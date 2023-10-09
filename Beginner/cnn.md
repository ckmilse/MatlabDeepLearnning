
 
 卷积层 

 池化层

 修正线性单元层


  串行网络（串联）

 有向无循环网络（DAG）

 层数组


### 提高性能
```
load pathToImages
load trainedFlowerNetwork
flwrds = imageDatastore(pathToImages,"IncludeSubfolders",true,"LabelSource","foldernames");
[trainImgs,testImgs] = splitEachLabel(flwrds,0.99);
resizeTestImgs = augmentedImageDatastore([224 224],testImgs);
flwrPreds = classify(flowernet,resizeTestImgs);

flwrActual = testImgs.Labels
<!-- 您可以使用逻辑比较和 nnz 函数来确定两个数组的匹配元素数：
numequal = nnz(a == b) -->

numCorrect = nnz(flwrPreds == flwrActual)

<!-- 通过将 numCorrect 除以测试图像的数量，计算正确分类的测试图像的比例。将结果存储在名为 fracCorrect 的变量中。 -->
fracCorrect = numCorrect/numel(flwrPreds)

<!-- confusionchart 函数计算并显示预测分类的混淆矩阵 -->
confusionchart(flwrActual,flwrPreds)

<!-- 要研究误分类的图像，您可以找到哪些文件包含误分类的图像并查看这些图像。例如，以下代码显示第二个误分类的图像 -->
idxWrong = find(flwrPreds ~= flwrActual)
idx = idxWrong(1)
imshow(readimage(testImgs,idx))
title(testImgs.Labels(idx))
```



