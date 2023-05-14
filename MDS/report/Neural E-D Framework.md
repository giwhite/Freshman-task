# Neural E-D Framework

任务：

**<span style="background-color: rgb(255, 255, 255)">单文档摘要</span>**<span style="background-color: rgb(255, 255, 255)">从给定的一个文档中生成摘要，</span>**<span style="background-color: rgb(255, 255, 255)">多文档摘要</span>**<span style="background-color: rgb(255, 255, 255)">从给定的一组主题相关的文档中生成摘要</span>

### 模型的优点

1.  该方法鲁棒性强，不需要MDS训练数据

2.  它从巨型文档中识别出少数最重要的句子

### 使用的方法

1.  我们介绍了一种结合指针生成器( PG )网络和最大边缘相关性( MMR )算法的自适应方法。

2.  从多文档输入中生成摘要的一个有前途的途径是将为单文档摘要训练的神经编码器-解码器模型应用于测试时将集合中的所有文档串联创建的"大文档"，

    训练的策略就是使用但文本的摘要，通过但文本中学习的内容，放到多文本摘要中

3.  The attention weights of the PG model are directly modified to focus on these important sentences when generating a summary sentence

    使用了最大边缘概率的模型



### PG+MMR

1. MMR是一个”external”model，这个model是用来解决mult DOCS的问题通过执行文本抽象和句子融合生成文档摘要


2. 对PG模型进行训练，有效识别输入文档中的重要内容

#### context vector

1. 构造了一个上下文向量( ct ) ( Eq . ( 4 ) )总结输入的语义,编码器隐藏状态的加权和


<img align="centering" src="attachments/MG4BLA27.png" style="zoom:50%;" />\

2. 注意力权重α t，i衡量了第i个输入词对生成第t个输出词的重要性

​		一层的attention\_weight，然后把它和每一次的hidden\_state，进行乘积，得到这个文本的表征向量，也就是context vector，然后		放	到一个linear层，这个linear层需要加上偏置。和decoder中的隐藏层进行乘积，然后放到一个softmax层中得到输出

#### MMR的具体实现

<img src=".\attachments\IY5RYCEH.png" alt="IY5RYCEH" style="zoom:50%;" />

注意此时都是在讲解test的阶段的情况，这些情况下使用的MMR

1.  问题：

    - 从多文档输入中识别显著性内容

    
    - 其次，注意力机制是基于输入词的位置而不是其语义
    
2.  解决多文本问题：

    - 使用MMR模型，在这个PG模型的外部，来得到最大关联性的句子。

3.  想法：

    - 通过调整网络的注意力值，将最大边缘相关性算法

4.  MMR处理方式：

    - MMR从文档( D )中选择一个句子，并将其包含在摘要( S )中，直到达到长度阈值

    
- Sim1( si , D)衡量句子si与文档的相似度。它充当句子重要性的代理🔤
  
    <img src="attachments/UTA4H9I5.png" style="zoom:63%;" />
    
    ***
    

PG - MMR遵循MMR原则选择得分最高的K个源句

##### **Muting方式**

> 这个方式是挑出选中的句子，然后将这个句子的每个此词的attention\_weight，然后其他句子的attention\_weight清零。然后此时句子需要重新被归一化

##### **Sentence Importance.**

1.  这个是需要被训练出来的模型

    重要的是，该模型在训练数据丰富的单文档摘要数据集上进行训练，该模型的意义是可用于从多文档输入中识别重要句子。

    ***

    1.  sentence length

    2.  its absolute and relative position in the document

    3.  sentence quality

        通过PG模型来构建句子的表示，使用的是LSTM双向编码器。

    4.  how close the sentence is the main topic of the document set


    ***

2.  计算方式：

    - 首先需要通过PG模型（我们就是BART模型）得到这个句子的represent，也就是BART模型的输出？


    - 然后通过LSTMencoder来使用前后隐藏层的cat，也就是`source sentence`的`vector representation`


    - 然后计算得到document vector，文档向量是所有句子向量的平均值


    - 我们使用文档向量和文档与句子向量之间的余弦相似度作为指标

3.  训练的方式

    在(句子,得分)对上训练支持向量回归模型，其中训练数据来自CNN / Daily Mail数据集，评分需要几个句子的特征来表明

    

##### Sentence Redundancy

- 计算这个是通过ROUGE-L的精度来说明两个的句子的冗余

  ROUGEL精度较高的源语句被认为与部分摘要有显著的内容重叠，此时就会给他一个更低的MMR分数，这样这个句子就会被相对的忽略，相似的程度通过余弦相似度来表示

### 算法的流程：

1.  首先通过PG模型计算出来summary sentence
2.  更新MMR scores -> source sentence
3.  修改之后这个source sentences 集（猜测是一个abstract集）会引导PG去生产下一个summary sentence



