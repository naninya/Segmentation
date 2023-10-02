# 紹介
VIA（VGG Image Annotator）、SMP（segmentation_models.pytorch）をBenchmarkし、ラベル生成を含んだSegmentationの全体プロセスを行える

# docker 構造
via - VGG Image Annotator  
smp　- segmentation_models.pytorch  

# 起動手順
1. git clone  
2. (./segmentation のdirectoryに移動)
3. $docker-compose up  （GPU環境ではないとDockerが起動しません。）
4. {domain}:8021に接続し、Segmentationを行なってから.jsonファイルをDownloadする  
(Segmentationラベルがあったらスルー)  
4. データを入れる  
(1)./smp/data/train/{project_name}/images --> 学習イメージ  
(2)./smp/data/train/{project_name}/annotations --> セグメンテーションマスク or ./smp/data/train/??.json (viaの結果物)  
(3)./smp/data/test/{project_name}/images --> テストイメージ  
(4)./smp/data/test/{project_name}/annotations --> セグメンテーションマスク or ./smp/data/test/??.json (viaの結果物)  
5. 学習、テスト  
*command lineで行う場合   
(1) docker exec -it smp .bash  
(2) python ./src/train.py  
(3) python ./src/test.py  
*notebookで行う場合  
(1) {domain}:8022に接続
(2) ./smp/notebook/Sample_mvtec.ipynb を実行  
6. 実行結果  
./smp/result/{project_name}/{scheme}.ckpt --> 学習Model weight  
./smp/result/{project_name}/result.pkl --> テストデータの結果  
./smp/result/{project_name}/segmentation_image/{scheme}/*.png --> テスト結果イメージ  
