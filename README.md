# IFT6135_Winter2019_Assignement2
Assignement 2

Team members:
* Manal Maroua BOUCENNA
* Andres Lou
* Mirmohammad Saadati
* Hadi Abdi

## Problem 4
### 4.1
### 4.2
### 4.3
In this part, we conduct a hyperparameter search with each model (_RNN_, _GRU_, and _TRANSFORMER_). The following tables illustrates tested configurations and corresponding results.

#### RNN
|Model|Optimizer|Initial LR|Batch Size|Seq Len|Hidden Size|Num Layers|DP Keep|Train Last PPL|Valid Last PPL|Valid Best PPL|
|-----|---------|----------|----------|-------|-----------|----------|-------|--------------|--------------|--------------|
|RNN|ADAM|0.0001|20|15|1500|2|0.35|121.2331579|177.5850408|172.2200323|
|RNN|ADAM|0.0001|20|50|1500|2|0.35|127.0711768|155.8327936|155.8327936|
|RNN|ADAM|0.0001|20|35|750|2|0.35|142.7558341|158.192536|157.8210037|
|RNN|ADAM|0.0001|20|35|3000|2|0.35|107.8629854|166.8691057|163.894203|
|RNN|ADAM|0.0001|20|35|1500|1|0.35|114.5198958|152.6916274|152.6916274|
|RNN|ADAM|0.0001|20|35|1500|4|0.35|148.2102388|187.788502|183.5583428|

#### GRU
|Model|Optimizer|Initial LR|Batch Size|Seq Len|Hidden Size|Num Layers|DP Keep|Train Last PPL|Valid Last PPL|Valid Best PPL|
|-----|---------|----------|----------|-------|-----------|----------|-------|--------------|--------------|--------------|
|GRU|SGD + Scheduler|10|20|17|1500|2|0.35|68.40980413|104.1473476|104.1472937|
|GRU|SGD + Scheduler|10|20|70|1500|2|0.35|72.16150674|104.296174|104.2961653|
|GRU|SGD + Scheduler|10|20|35|750|2|0.35|85.39954015|104.3537161|104.3536967|
|GRU|SGD + Scheduler|10|20|35|3000|2|0.35|52.81547796|106.1929572|106.1929316|
|GRU|SGD + Scheduler|10|20|35|1500|1|0.35|59.04505332|95.50512491|95.5051184|
|GRU|SGD + Scheduler|10|20|35|1500|3|0.35|75.73174797|112.6179822|112.6179536|

#### TRANSFORMER
|Model|Optimizer|Initial LR|Batch Size|Seq Len|Hidden Size|Num Layers|DP Keep|Train Last PPL|Valid Last PPL|Valid Best PPL|
|-----|---------|----------|----------|-------|-----------|----------|-------|--------------|--------------|--------------|
|Transformer|SGD + Scheduler|20|128|20|512|6|0.9|26.28393856|198.9976264|160.8568172|
|Transformer|SGD + Scheduler|20|128|70|512|6|0.9|177.4911996|215.4643363|215.4643363|
|Transformer|SGD + Scheduler|20|128|35|256|6|0.9|88.4081457|142.3776466|142.3776424|
|Transformer|SGD + Scheduler|20|128|35|1024|6|0.9|34.88865654|166.2080425|155.9741435|
|Transformer|SGD + Scheduler|20|128|35|512|4|0.9|64.75014313|152.4738212|152.4737894|
|Transformer|SGD + Scheduler|20|128|35|512|8|0.9|67.86015482|138.9884592|138.9884344|
