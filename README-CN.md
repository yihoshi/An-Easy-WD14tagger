该项目在以KutsuyaYuki/WD14Tagger为基础进行修改

Python：推荐使用3.10.9

安装依赖
项目包含了一个setup.bat脚本，用于创建虚拟环境并安装所有必要的依赖项。
国内网络不佳可以使用阿里云镜像加速

参数设置：run_tagger.bat中的参数可以通过编辑脚本来进行调整。
主要参数包括：
outputDir：处理后图片的输出目录。
modelDir：ONNX模型的目录。
batchSize：批处理大小。
threshold：置信度阈值。
captionExtension：生成的标签文件扩展名。
replaceUnderscores：是否将标签中的下划线替换为空格。
maxDataLoaderWorkers：数据加载器的最大线程数。

使用run_tagger.bat调用tagger.py进行图片打标
