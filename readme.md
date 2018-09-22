&emsp;&emsp;这是Windows下caffe版SSD的实现，SSD克隆自[WeiLiu](https://github.com/weiliu89/caffe/tree/ssd)，已经更新到最新版（截至到2018-05-31），caffe基于微软caffe。此库支持SSD以及其他caffe程序，其中我还加入了一些新的层，包括 ***focal_loss_layer***, ***DepthWise Convolution***, ***Relu6***, ***Selu***. 为了使仓库体积最小，我删掉了一些不是那么重要的文件和文件夹，例如examples文件夹，如果需要使用，从BVLC的caffe库或者SSD复制过来即可。你也可以从[这里](https://pan.baidu.com/s/12lff-Mn8Jja1d1CMUkvtXA)下载我的完整的包win-ssd.zip，里面也已经包含了编译出的文件。				

### 开发环境		
&emsp;&emsp;我使用的开发环境为：win10，vs2013，anaconda2，opencv3.4。我的代码放置路径为 D:/win-ssd/，你的环境不需要完全和我一样，但为了出现更少的编译错误，还是建议和我保持一样。下面重点说一下anaconda和opencv。anaconda必须是2，因为caffe需要的python版本为2.7，opencv的版本没有限制，但我认为你应该和我一样更愿意用比较新的版本。anaconda和opencv安装完后必须加入到环境变量，不然编译caffe的时候会找不到依赖库。			

### 使用			
&emsp;&emsp;如果你只需要使用现有的caffe，不需要修改相关代码（例如添加新的层），那你可以直接使用我已经编译好的文件，从[这里](https://pan.baidu.com/s/12lff-Mn8Jja1d1CMUkvtXA)下载Release.zip文件。			
&emsp;&emsp;首先clong整个文件夹，放置在D盘下，并将主目录重命名为win-ssd。然后把下载的Release.zip解压放置到caffe-ssd-ms下，并将Release加入到环境变量。接着你就可以开发你自己的程序了。如果你开发的是C++程序，可以参考[CaffeTemplate](http://)，里面是已经建立好的独立的caffe工程。如果你开发的是python程序，你需要将Release/pycaffe/caffe文件夹放置于anaconda2/lib/site-packages文件夹下，然后你就可以像在ubuntu下那样使用caffe。如果你更喜欢使用matlab版本的caffe，请自行编译。			

### 编译			
&emsp;&emsp;1、必须安装好anaconda2和opencv（如果不使用opencv可以不安装）。（1）从[这里](https://pan.baidu.com/s/12lff-Mn8Jja1d1CMUkvtXA)下载Nuget依赖包。你也可以在编译的时候开启Nuget还原，自动下载依赖包，不过很慢，甚至根本下载不了。（2）下载上面网盘里的windows.zip文件，解压后放置于caffe-ssd-ms文件夹下。文件目录：				
D:/win-ssd/			
&emsp;&emsp;--caffe-ssd-ms
&emsp;&emsp;&emsp;&emsp;--windows
&emsp;&emsp;--NugetPackages					
&emsp;&emsp;2、找到caffe-ssd-ms/windows/caffe.sln，使用vs2013打开。在解决方案窗口，打开props/CommonSettings.props，里面是工程的一些配置选项。比较重要的选项有CPU、GPU、CUDNN开关，CUDA版本，cudnn路径，python目录（anaconda2路径），默认使用opencv。根据我已经配置好的路径适当修改为自己的路径。				
&emsp;&emsp;3、在调试->属性选择Release、x64，在生成->配置管理器勾选需要编译的项目。			
&emsp;&emsp;4、接下来就是编译了，时间比较长。			
&emsp;&emsp;5、这个代码是我编译了将近一个星期才完成的，解决了N多错误。自己也已经用它做过了几个项目，所以代码是没有问题的，如果编译当中出现了错误，一般是路径的问题，请对照提示信息，仔细检查。如果仍然有问题，可以从我上面给出的链接直接下载完整的包编译，也可以给我发邮件（zxdzhuwei@foxmail.com）。