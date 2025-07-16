2.1注意力机制

    1.从计算机视觉（Computer Vision，CV）为起源发展起来的神经网络，其核心架构有三种：
        1.前馈神经网络（Feedforward Neural Network，FNN），即每一层的神经元都和上下两层的每一个神经元完全连接
        2.卷积神经网络（Convolutional Neural Network，CNN），即训练参数量远小于前馈神经网络的卷积层来进行特征提取和学习
        3.循环神经网络（Recurrent Neural Network，RNN），能够使用历史信息作为输入、包含环和自重复的网络

    2.注意力机制概念
        注意力机制最先源于计算机视觉领域，其核心思想为当我们关注一张图片，我们无需看清楚全部内容而仅将注意力集中在重点部分即可。而在自然语言处理领域，我们往往也可以通过将重点注意力集中在一个或几个 token，从而取得更高效高质的计算效果。（抓住重点放弃非重点）

        注意力机制有三个核心变量：查询值 Query，键值 Key 和 真值 Value。接下来我们以字典为例，逐步分析注意力机制的计算公式是如何得到的：
            "apple":10,
            "banana":5,
            "chair":2
        例如，当我们的 Query 为“fruit”，我们可以分别给三个 Key 赋予如下的权重：
            "apple":0.6,
            "banana":0.4,
            "chair":0
        那么，我们最终查询到的值应该是：
            value=0.6∗10+0.4∗5+0∗2=8

    3.自注意力
        注意力机制的本质是对两段序列的元素依次进行相似度计算，寻找出一个序列的每个元素对另一个序列的每个元素的相关度，然后基于相关度进行加权，即分配注意力。

        在我们的实际应用中，我们往往只需要计算 Query 和 Key 之间的注意力结果，很少存在额外的真值 Value。也就是说，我们其实只需要拟合两个文本序列。​在经典的注意力机制中，Q 往往来自于一个序列，K 与 V 来自于另一个序列，都通过参数矩阵计算得到，从而可以拟合这两个序列之间的关系。

    4.掩码自注意力
        掩码自注意力，即 Mask Self-Attention，是指使用注意力掩码的自注意力机制。掩码的作用是遮蔽一些特定位置的 token，模型在学习的过程中，会忽略掉被遮蔽的 token。
        使用注意力掩码的核心动机是让模型只能使用历史信息进行预测而不能看到未来信息。

        例如，如果待学习的文本序列是 【BOS】I like you【EOS】，那么，模型会按如下顺序进行预测和学习：
        Step 1：输入 【BOS】，输出 I
        Step 2：输入 【BOS】I，输出 like
        Step 3：输入 【BOS】I like，输出 you
        Step 4：输入 【BOS】I like you，输出 【EOS】

        如果对于每一个训练语料，模型都需要串行完成上述过程才能完成学习，那么很明显没有做到并行计算，计算效率很低。
        针对这个问题，Transformer 就提出了掩码自注意力的方法。掩码自注意力会生成一串掩码，来遮蔽未来信息。例如，我们待学习的文本序列仍然是 【BOS】I like you【EOS】，我们使用的注意力掩码是【MASK】，那么模型的输入为：
        <BOS> 【MASK】【MASK】【MASK】【MASK】
        <BOS>    I   【MASK】 【MASK】【MASK】
        <BOS>    I     like  【MASK】【MASK】
        <BOS>    I     like    you  【MASK】
        <BOS>    I     like    you   </EOS>
        观察上述的掩码，我们可以发现其实则是一个和文本序列等长的上三角矩阵。我们可以简单地通过创建一个和输入同等长度的上三角矩阵作为注意力掩码，再使用掩码来遮蔽掉输入即可。
        在注意力计算时，我们会将计算得到的注意力分数与这个掩码做和，上三角区域（也就是应该被遮蔽的 token 对应的位置）的注意力分数结果都变成了 -inf，而下三角区域的分数不变。再做 Softmax 操作，-inf 的值在经过 Softmax 之后会被置为 0，从而忽略了上三角区域计算的注意力分数，从而实现了注意力遮蔽。

    5.多头注意力
        Transformer 使用了多头注意力机制（Multi-Head Attention），即同时对一个语料进行多次注意力计算，每次注意力计算都能拟合不同的关系，将最后的多次结果拼接起来作为最后的输出，即可更全面深入地拟合语言信息。

2.2Encoder-Decoder

    在本节中，我们将以上一节所介绍的 注意力机制为基础，从 Transformer 所针对的 Seq2Seq 任务出发，解析 Transformer 的 Encoder（编码器）-Decoder（解码器）结构。

    1.Seq2Seq模型
        序列到序列，是一种经典 NLP 任务，几乎所有的 NLP 任务都可以视为 Seq2Seq 任务。
        对于 Seq2Seq 任务，一般的思路是对自然语言序列进行编码再解码。编码：将输入的自然语言序列通过隐藏层编码成能够表征语义的向量（或矩阵），可以简单理解为更复杂的词向量表示。
        解码：将输入的自然语言序列编码得到的向量或矩阵通过隐藏层输出，再解码成对应的自然语言目标序列。
        Transformer 由 Encoder 和 Decoder 组成，每一个 Encoder（Decoder）又由 6个 Encoder（Decoder）Layer 组成。

    2.前馈神经网络（FNN）
        每一个 Encoder Layer 都包含一个上文讲的注意力机制和一个前馈神经网络。
        Transformer 的前馈神经网络是由两个线性层中间加一个 RELU 激活函数组成的，以及前馈神经网络还加入了一个 Dropout 层来防止过拟合。

    3.层归一化
        神经网络主流的归一化一般有两种，批归一化（Batch Norm）和层归一化（Layer Norm）。
        归一化核心是为了让不同层输入的取值范围或者分布能够比较一致。

        在深度神经网络中，需要归一化操作，将每一层的输入都归一化成标准正态分布。批归一化是指在一个 mini-batch 上进行归一化，相当于对一个 batch 对样本拆分出来一部分。
        批归一化存在一些缺陷：
            1.当显存有限，mini-batch 较小时，取的样本的均值和方差不能反映全局的统计分布信息，从而导致效果变差；
            2.对于在时间维度展开的 RNN，Batch Norm 的归一化会失去意义；
            3.在训练时，Batch Norm 需要保存每个 step 的统计信息（均值和方差）。在测试时，由于变长句子的特性，测试集可能出现比训练集更长的句子，所以对于后面位置的 step，是没有训练的统计量使用的；
            4.应用 Batch Norm，每个 step 都需要去保存和计算 batch 统计量，耗时又耗力

        因此，出现了在深度神经网络中更常用、效果更好的层归一化（Layer Norm）。Layer Norm 的归一化方式其实和 Batch Norm 是完全一样的，只是统计统计量的维度不同。

    4.残差链接
        为了避免模型退化，Transformer 采用了残差连接的思想来连接每一个子层。残差连接，即下一层的输入不仅是上一层的输出，还包括上一层的输入。残差连接允许最底层信息直接传到最高层，让高层专注于残差的学习。

    5.Encoder
        Encoder 由 N 个 Encoder Layer 组成，每一个 Encoder Layer 包括一个注意力层和一个前馈神经网络。因此，我们可以首先实现一个 Encoder Layer。
        然后我们搭建一个 Encoder，由 N 个 Encoder Layer 组成，在最后会加入一个 Layer Norm 实现规范化。

    6.Decoder
        类似的，我们也可以先搭建 Decoder Layer，再将 N 个 Decoder Layer 组装为 Decoder。
        不同的是，Decoder 由两个注意力层和一个前馈神经网络组成。
        第一个注意力层是一个掩码自注意力层，即使用 Mask 的注意力计算，保证每一个 token 只能使用该 token 之前的注意力分数；第二个注意力层是一个多头注意力层，该层将使用第一个注意力层的输出作为 query，使用 Encoder 的输出作为 key 和 value，来计算注意力分数。
        最后，再经过前馈神经网络。

2.3搭建一个Transformer

    1.Embedding 层
        在 NLP 任务中，我们往往需要将自然语言的输入转化为机器可以处理的向量。在深度学习中，承担这个任务的组件就是 Embedding 层。
        在输入神经网络之前，我们往往会先让自然语言输入通过分词器 tokenizer，分词器的作用是把自然语言输入切分成 token 并转化成一个固定的 index。
        exp.input: 我   output: 0
            input: 喜欢   output: 1
            input：你   output: 2
        Embedding 层的输入往往是一个形状为 （batch_size，seq_len，1）的矩阵，第一个维度是一次批处理的数量，第二个维度是自然语言序列的长度，第三个维度则是 token 经过 tokenizer 转化成的 index 值。

    2.位置编码
        在注意力机制的计算过程中，对于序列中的每一个 token，其他各个位置对其来说都是平等的，即“我喜欢你”和“你喜欢我”在注意力机制看来是完全相同的。为使用序列顺序信息，保留序列中的相对位置信息，Transformer 采用了位置编码机制。
        位置编码，即根据序列中 token 的相对位置对其进行编码，再将位置编码加入词向量编码中。

    3.一个完整的Transformer
        class Transformer(nn.Module):
            '''整体模型'''
            def __init__(self, args):
                super().__init__()
                # 必须输入词表大小和 block size
                assert args.vocab_size is not None
                assert args.block_size is not None
                self.args = args
                self.transformer = nn.ModuleDict(dict(
                    wte = nn.Embedding(args.vocab_size, args.n_embd),
                    wpe = PositionalEncoding(args),
                    drop = nn.Dropout(args.dropout),
                    encoder = Encoder(args),
                    decoder = Decoder(args),
                ))
                # 最后的线性层，输入是 n_embd，输出是词表大小
                self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

                # 初始化所有的权重
                self.apply(self._init_weights)

                # 查看所有参数的数量
                print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

            '''统计所有参数的数量'''
                def get_num_params(self, non_embedding=False):
                    # non_embedding: 是否统计 embedding 的参数
                    n_params = sum(p.numel() for p in self.parameters())
                    # 如果不统计 embedding 的参数，就减去
                    if non_embedding:
                        n_params -= self.transformer.wpe.weight.numel()
                    return n_params

                '''初始化权重'''
                def _init_weights(self, module):
                    # 线性层和 Embedding 层初始化为正则分布
                    if isinstance(module, nn.Linear):
                        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.Embedding):
                        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
                '''前向计算函数'''
                def forward(self, idx, targets=None):
                    # 输入为 idx，维度为 (batch size, sequence length, 1)；targets 为目标序列，用于计算 loss
                    device = idx.device
                    b, t = idx.size()
                    assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"

                    # 通过 self.transformer
                    # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
                    print("idx",idx.size())
                    # 通过 Embedding 层
                    tok_emb = self.transformer.wte(idx)
                    print("tok_emb",tok_emb.size())
                    # 然后通过位置编码
                    pos_emb = self.transformer.wpe(tok_emb) 
                    # 再进行 Dropout
                    x = self.transformer.drop(pos_emb)
                    # 然后通过 Encoder
                    print("x after wpe:",x.size())
                    enc_out = self.transformer.encoder(x)
                    print("enc_out:",enc_out.size())
                    # 再通过 Decoder
                    x = self.transformer.decoder(x, enc_out)
                    print("x after decoder:",x.size())

                    if targets is not None:
                        # 训练阶段，如果我们给了 targets，就计算 loss
                        # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
                        logits = self.lm_head(x)
                        # 再跟 targets 计算交叉熵
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                    else:
                        # 推理阶段，我们只需要 logits，loss 为 None
                        # 取 -1 是只取序列中的最后一个作为输出
                        logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
                        loss = None

                    return logits, loss
        上述代码除去搭建了整个 Transformer 结构外，我们还额外实现了三个函数：
            get_num_params：用于统计模型的参数量
            _init_weights：用于对模型所有参数进行随机初始化
            forward：前向计算函数
        经过上述步骤，我们就可以从零“手搓”一个完整的、可计算的 Transformer 模型


