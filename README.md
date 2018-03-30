# PhotoStyleTransfer


### 图像风格转换
这个项目是我读完Leon A. Gatys等的“A Neural Algorithm of Artistic Style”以及李飞飞的“Perceptual Losses for Real-Time Style Transfer and Super-Resolution”后的一个练手小项目。
学生党的玩（com）具（puter）配置太垃圾了，所以就选了一个比较对显卡友好的方案来实现简单的图像风格迁移。

### 方案
Google的TensorFlow框架；
卷积神经网络，使用的是vgg16预训练模型，[下载vgg16预训练模型](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM);
输入提供的风格图片、提供内容的图片，以及目标图片。为了加快训练，目标图片=内容图片+噪声+高斯模糊

### 结果
##### 内容图片
![content](https://github.com/Ysjshine/PhotoStyleTransfer/blob/master/content.png?raw=true)
##### 风格图片
![style](https://github.com/Ysjshine/PhotoStyleTransfer/blob/master/style.png?raw=true)
##### 输出
由于电脑配置太垃圾了，所以只迭代了100次，不过还是看得出一点模样的
![output](https://github.com/Ysjshine/PhotoStyleTransfer/blob/master/out.png?raw=true)
