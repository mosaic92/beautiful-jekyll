---
layout: post
title: " Seasonal-ARIMA模型实例"
date: 2016-05-30
categories: blog
tags: [R,ARIMA,seasonal]

---

## 案例介绍

本文所使用的是一个非常著名的时间序列——航空客运量数据（Airline Passenger Data）。收集到的数据范围是从1949年1月到1960年12月间每个月的国际航空客运量，总共12年144个观测值。这是战后高速发展的时期，交通能力以及国际交流都在不断加强，从这个数据中可以发现有很明显的季节波动，还有明显的增长趋势，这给数据的分析预测带来了更复杂的干扰因素，如果使用时间序列的方法，要在序列的分析中加入季节因素和增长趋势。这个时间序列的统计特性随着时间的平移而发生变化，这就是前面提到的非平稳时间序列。现在，我们想预测未来的航空客运量的发展趋势，下面我们希望建立一个时序模型，这个模型的可以比较精确的反映国际航空客运量的规律，并借助这个模型来对未来进行预测。

## 模型定义

普通ARIMA模型适合具有长期趋势的时间序列，但是我们的案例中的航空客运量为月度数据，具有非常明显的季节变化，因此下面我们将再介绍Seasonal ARIMA模型：

$$
\Phi \left ( B^{12} \right )\left ( 1-B^{12} \right )^{D}X_{t}=\Theta \left ( B^{12} \right )a_{t}
$$

其中：

$$
\Phi \left(B \right )=1-\Phi_{1}B^{12}-\Phi_{2}B^{24}\cdots \Phi_{P}B^{12P}
$$

$$
\Theta \left(B \right )=1-\Theta_{1}B^{12}-\Theta_{2}B^{24}\cdots \Theta_{P}B^{12P}
$$

该模型简记为：
$$
ARIMA\left(P,D,Q \right )_{12}
$$
这个模型和
$$
ARIMA\left(p,d,q \right )
$$
看起来很相似，不过是将B转化为B的12次方，凸显以12个月为周期的季节变化。

在季节变化的时间序列中更常用混合的ARIMA模型： 

$$
\Phi\left(B^{12} \right )\phi\left(B \right )\left(1-B \right )^{d} \left(1-B^{12} \right)^{D}X_{t}=\theta\left(B \right )\Theta\left(B^{12} \right )a_{t}
$$

该模型记为：

$$
ARIMA\left(p,d,q \right )\times \left(P,D,Q \right )_{12}
$$

下面我们将用混合的ARIMA模型来拟合国际航空客运量的时间序列模型，在以下的步骤中，将会识别出季节变化和长期趋势变化。

## 描述性统计

首先我们还是对该数据进行描述性的分析，通过时间序列图来观察变量的变化情况。 

```R
library(tseries)
passenger <- read.csv("passenger.csv")
a <- ts(passenger$X, start=c(1949,1),frequency = 12)
plot(a, type="l", main="Airline Data")
```

 ![Air1](https://raw.githubusercontent.com/mosaic92/mosaic92.github.io/master/img/airpassage/Air1.jpeg)

首先我们来看看国际航空客运量在这12年里的变化情况。从图中可以看到几个非常明显的趋势。首先，客运量总体上呈现上升趋势，在战后随着经济的复苏以及全球化浪潮的影响越来越大，国际航空客运行业也越来越发达；其次是季节性的趋势，以一年十二个月为周期，航空客运量呈现周期变化的趋势，每年都拥有相似的高峰期和低谷期；最后是波动加剧的趋势，随着时间推移，每年不同月份的国际航空客运量的差异越来越大。
根据以上时间序列图反映的长期上升趋势、季节趋势和波动加剧的趋势，显然我们可以认定这个时间序列是非平稳时间序列。首先为了波动加剧的趋势（也就是时间序列的方差在不断加大），使每年的国际航空客运量的变化更加相近，我们使用取对数的方法。

```R
plot(log(a), type="l", main="Airline Data")
```

 ![Air2](https://raw.githubusercontent.com/mosaic92/mosaic92.github.io/master/img/airpassage/Air2.jpeg)

观察图形，在取过对数后季节波动幅度不断增大的趋势被抑制了，将每年的数据进行比较，可以发现有很强的相似性，似乎就是将某一年的数据进行了纵向的平移得到的。尽管如此，之前的季节趋势和增长趋势仍是存在的，该时间序列仍然是非平稳时间序列。为了消除季节趋势和增长趋势，我们需要对该序列进行差分。那么究竟应该进行几阶的差分才可以达到我们的目的呢？这就涉及到了模型识别和定阶的内容了。



## 模型识别

由上面内容可知，国际航空客运量序列是一个既含有季节效应又含有长期上升趋势的时间序列。序列的季节效应、长期趋势效应和随机波动之间存在复杂的交互影响关系，若简单的进行几阶差分并不足以提取其中的相关关系，这里我们考虑乘积季节模型，对其进行1阶12步差分。

```R
lga.diff <- diff(diff(log(a)),12)
plot(lga.diff,type='l')
```

 ![Air3](https://raw.githubusercontent.com/mosaic92/mosaic92.github.io/master/img/airpassage/Air3.jpeg)

绘制差分后序列自相关图和偏自相关图

```R
acf(lga.diff,main="ACF")
pacf(lga.diff,main="PACF")
```

 ![air4](https://raw.githubusercontent.com/mosaic92/mosaic92.github.io/master/img/airpassage/air4.jpeg)

 ![air5](https://raw.githubusercontent.com/mosaic92/mosaic92.github.io/master/img/airpassage/air5.jpeg)

首先考虑1阶12步差分之后，序列12阶以内的自相关系数和偏自相关系数的特征，以确定短期相关模型。自相关图和偏自相关图显示12阶以内的自相关系数和偏自相关系数均不截尾，所以尝试使用
$$
ARMA(1,1)
$$
模型提取差分后序列的短期自相关信息。
再考虑季节自相关的特征，这是考虑延迟12阶、24阶等以周期长度为单位的自相关系数和偏自相关系数的特征。自相关图和偏自相关图均显示延迟12阶自相关系数显著非零，但是延迟24阶自相关系数落入2倍标准差范围。这时以12步为周期的
$$
ARMA(1,1)_{12}
$$
模型提取差分后序列的季节相关信息。
综合前面的差分信息，我们要拟合的乘积模型为：
$$
ARIMA\left(1,1,1 \right )\times \left(1,1,1 \right )_{12}
$$

## 参数估计

```R
a.fit <- arima(a,order=c(1,1,1),seasonal=list(order=c(1,1,1),period=12))

a.fit
Call:
arima(x = a, order = c(1, 1, 1), seasonal = list(order = c(1, 1, 1), period = 12))

Coefficients:
          ar1      ma1     sar1    sma1
      -0.1386  -0.2028  -0.9228  0.8329
s.e.   0.5865   0.6128   0.2387  0.3519

sigma^2 estimated as 130.8:  log likelihood = -506.15,  aic = 1022.3

```

得到的拟合模型为：

$$
X_{t}=\frac{(1-0.2028B)(1+0.8329B^{12})}{(1+0.1386B)(1+0.9228B^{12})}a_{t},a_{t}\sim N(0,130.8)
$$

## 模型检验

使用Box.test函数对模型进行白噪声检验，每次调用Box.test函数时，只能给出一个检验结果。为了得到延迟6阶和12阶两个LB统计量的结果，我们做了循环。

```R
for(i in 1:2) print (Box.test(a.fit$residuals,lag=6*i))

    Box-Pierce test

data:  a.fit$residuals
X-squared = 4.6005, df = 6, p-value = 0.596


	Box-Pierce test

data:  a.fit$residuals
X-squared = 10.085, df = 12, p-value = 0.6085
```

P值大于0.01，所以该序列不能拒绝纯随机的原假设，说明该模型显成立，ARIMA(1,1,1)*(1,1,1)12 模型对该序列拟合成功。

## 模型预测

使用forecast函数以得到的模型对1961~1962年的国际航空客运量进行预测并作图

```R
library(forecast)
a.fore <- forecast(a.fit,h=12)
plot(a.fore, main='Forecast')

         Point Forecast    Lo 80    Hi 80    Lo 95    Hi 95
Jan 1961       449.2235 434.5558 463.8913 426.7911 471.6560
Feb 1961       424.3086 406.7467 441.8705 397.4500 451.1672
Mar 1961       459.0040 438.6181 479.3899 427.8264 490.1816
Apr 1961       497.7232 474.9029 520.5436 462.8225 532.6239
May 1961       509.7475 484.7230 534.7720 471.4759 548.0191
Jun 1961       568.2004 541.1515 595.2493 526.8327 609.5681
Jul 1961       655.7378 626.8057 684.6699 611.4899 699.9856
Aug 1961       641.1741 610.4741 671.8741 594.2225 688.1257
Sep 1961       546.3759 514.0044 578.7473 496.8680 595.8838
Oct 1961       496.7516 462.7908 530.7124 444.8130 548.6901
Nov 1961       427.6982 392.2193 463.1772 373.4378 481.9587
Dec 1961       471.2936 434.3587 508.2284 414.8065 527.7806

```

 ![air6](https://raw.githubusercontent.com/mosaic92/mosaic92.github.io/master/img/airpassage/air6.jpeg)

