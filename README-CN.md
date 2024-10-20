该项目在在KutsuyaYuki/WD14Tagger的基础上进行修改


Python：推荐使用3.10.9

安装依赖
运行setup.bat创建虚拟环境并安装依赖

或者

pip install -r requirements.txt

国内网络不佳可以使用阿里云镜像加速

参数设置：在运行前请编辑run_tagger.bat中的参数
主要参数包括：

outputDir：打标后inputdir会被移动到这里

modelDir：ONNX模型的目录。

batchSize：批处理大小。

threshold：置信度阈值。

captionExtension：生成的标签文件扩展名。

replaceUnderscores：是否将标签中的下划线替换为空格。

maxDataLoaderWorkers：数据加载器的最大线程数。

只支持onnx模型，且不提供下载模型的功能，请将模型和selected_tags.csv放入你指定的文件夹

运行run_tagger.bat调用tagger.py进行图片打标，打标完成后会将inputdir移入你指定的outputdir。

在Script completed successfully!后可以Ctrl+C退出脚本从而不移动文件夹。




