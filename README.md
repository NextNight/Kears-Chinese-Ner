# 基于keras+Bilstm+softmax/crf的中文命名实体识别
这是一个基于keras+Bilstm+softmax/crf的中文命名实体识别练习，主要是理解命名实体识别的做法以及bilstm,crf层。bilstm对于捕捉序列数据的长依
赖非常有效，而crf层主要是去学习实体间的状态依赖关系，学习到隐状态间的转移概率，拿BIO数据集来说，一个实体一定是BIIII..这种结构，不存在IIII这
种结构，也不存在BOIIIII这种结构，crf会学习到B之后只能接I,I的最前面必须有B.这是softmax层所学习不到的。
# 数据
- 采用人民日报BIO数据`B`表示实体的开头,`I`表示实体的其他部分，`O`表示非实体，`ORG`:表示组织实体，`PER`:表示人名，`LOC`:表示地名，格式如下
```
中 B-ORG
国 I-ORG
致 I-ORG
公 I-ORG
党 I-ORG
十 I-ORG
一 I-ORG
大 I-ORG
的 O
贺 O
词 O

各 O
位 O
代 O
表 O
、 O
各 O
位 O
同 O
志 O
： O

```
`\n\n`分割文档，`\n`分割字符和实体标签，


# 使用
- BILSTM_SOFTMAX_ZH_NER:以softmmax为最终层的实体识别。
- BILSTM_CRF_ZH_NER：以crf层为最终层的实体识别。

```bash
python BILSTM_SOFTMAX_ZH_NER.py
python BILSTM_CRF_ZH_NER.py
```

# 结果：
## BILSTM_CRF_ZH_NER结果：
- 输入1：
```
中华人民共和国国务院总理周恩来在外交部长陈毅，副部长王东的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚
```
- 输出：
```
PER: 周恩来 陈毅 王东
ORG: 中华人民共和国国务院 外交部
LOC： 埃塞俄比亚 非洲 阿尔巴尼亚
```

- 输入2：
```
中共中央批准，樊大志同志任中央纪委国家监委驻中国证券监督管理委员会纪检监察组组长，免去王会民同志的中央纪委国家监委驻中国证券监督管理委员会纪检监察组组长职务
```
- 输出：
```
PER: 樊大志 王会民
ORG: 中共中央 中央纪委国家监委驻 中国证券监督管理委员会纪检监察组 中央纪委国家监委驻 中国证券监督管理委员会纪检监察组
LOC：

```

- 输入3：
```
樊大志同志1987年8月在北京参加工作。先后在东北财经大学、北京市国有资产经营有限责任公司、北京证券有限责任公司、北京首都创业集团有限公司、华夏银行股份有限公司工作。
```
- 输出：
```
PER: 樊大志
ORG: 东北财经大学 北京市国有资产经营有限责任公司 北京证券有限责任公司 北京首都创业集团有限公司 华夏银行股份有限公司
LOC： 北京
```

由于最开始的例子太契合人民日报的内容了，所以有了后面几个示例，随机摘自网络新闻的数据。

# 存在的问题

- 数据截断：如果输入数据过长超过我们设置的max_len,就会导致结果很差，一个感觉就是，截断之后导致语义难以理解。
- 预处理：没有过滤低频词字
- 词向量：没有加入字符的EMbding
- 稍稍一训练准确率就达到1.0，吓得头皮发麻。。。。。

# 参考

- bilstm做NER的论文：[bilstm](./doc/1508.01991.pdf)
- crf:[crf-tutorial](./doc/crf-tutorial.pdf)
- tf做序列标注：[bi-LSTM + CRF](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)
- code:[https://github.com/stephen-v/zh-NER-keras](https://github.com/stephen-v/zh-NER-keras)

