# Word2vec implementation 
# 手刻 Word2vec模型
原論文 《Efficient Estimation of Word Representations in Vector Space》 https://arxiv.org/abs/1301.3781

## Word2vec自然語言模型
可將單詞轉成數學向量(vector)，或是想像成一個座標，並保留其詞與詞的關係。舉例來說，經過文章訓練後，"國王"之於"皇后"在空間中的距離，跟"王子"之於"公主"兩的距離一樣。

word2vec模型分為CWOB 和 Skip-gram 兩部分。
CWOB可以想像成將一個句子的某個單字挖空，透過前後文推測挖空的單詞。Ex. "失敗為__之母" __ = 成功機率最高
Skip-gram 相反可以想像成將一個單字，推測前後文的單詞。Ex. "寂寞" 前後出現 "空虛"、"覺得冷" 的機率最高

Word2vec 實作上有很多細節包括:負採樣、負採樣映射表等，
我也參考Airbnb做法、加上local negative、global context等針對商業邏輯的優化，以用於推薦系統。
目前還沒有看到網路上有這兩個中文版實作的文章，歡迎拿去修改交流。



參考文章: 
《word2vec中的数学原理详解》
https://blog.csdn.net/itplus/article/details/37969519

《word2vec-tutorial》
http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/

《霍夫曼樹實踐》
https://yunlongs.cn/2019/01/16/Word2vec%E4%B8%AD%E7%9A%84%E6%95%B0%E5%AD%A6%E5%8E%9F%E7%90%86%E8%AF%A6%E8%A7%A3/
《word2vec原理》
https://www.cnblogs.com/pinard/p/7160330.html
