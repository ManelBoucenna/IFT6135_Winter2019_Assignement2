# IFT6135_Winter2019_Assignement2
Assignement 2

Team members: Manal Maroua BOUCENNA, Andres Lou, Mirmohammad Saadati & Hadi Abdi.

## Problem 4
### 4.1
### 4.2
### 4.3
In this part, we conduct a hyperparameter search with each model (_RNN_, _GRU_, and _TRANSFORMER_). The following table illustrates tested configurations and corresponding results.

### RNN
|Model|Optimizer|Initial LR|Batch Size|Seq Len|Hidden Size|Num Layers|DP Keep|Train Last PPL|Valid Last PPL|Valid Best PPL|
|-----|---------|----------|----------|-------|-----------|----------|-------|--------------|--------------|--------------|
|RNN|ADAM|0.0001|20|15|1500|2|0.35|121.2331579|177.5850408|172.2200323|
|RNN|ADAM|0.0001|20|50|1500|2|0.35|127.0711768|155.8327936|155.8327936|
|RNN|ADAM|0.0001|20|35|750|2|0.35|142.7558341|158.192536|157.8210037|
