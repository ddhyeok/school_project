기말 발표자료, 


### school_project(파이썬 기반 딥러닝 코드 제출)



# git-구강이미지
  
  - method.py 파일에는 모델 학습, 검증, 시각화에 필요한 함수가 정의 되어있다.
    - train                   :분류모델 train
    - validate                :분류모델 validation
    - test                    :분류모델 test
    - model_train_eval        :분류모델 train, validation 후 test
    - CAE_train               :CAE train
    - CAE_ validate           :CAE validation
    - CAE_train_eval          :CAE train, validation
    - plot_loss_and_accuracy  :train, validation loss와accuracy plot+혼동행렬 plot
    - test_for_inference_time :inference time 측정을 위한 test
    
  
  - Grad-CAM.py 파일에는 Grad-CAM이 구현되어있다.
  
  - models 폴더에는 본 프로젝트에서 제안한 모델(CAE_v2, Classfier_v2)이 구현되어있다
  
  - Data_preprocess_split.ipynb 파일에서는 원천데이터를 전처리하고 split하여 저장한다.
  
    - Data/image_train_front,Data/label_train_front 폴더에 들어있는 원천데이터를 전처리한다.
    
    - Data/image_preprocessed_0,Data/image_preprocessed_1 폴더에 전처리된 이미지를 저정한다.
    
    - data를 split 하여 output_dataset 폴더 안에 저장한다. (각 데이터 셋의 클래스 비는 73.1(음성):26.9(양성)로 동일하다)
      
    - output_dataset

      ├── train ── 0,1

      ├── test ── 0,1

      └── val ── 0,1
    <img src=https://github.com/user-attachments/assets/729dd97f-de1d-44a8-a793-981f63f3a7f9 width="200" height="100">  
    <img src=https://github.com/user-attachments/assets/5488c939-63ce-4f03-805d-65b81e8350ee width="200" height="200">
    <img src=https://github.com/user-attachments/assets/c3e1283b-18c0-4a74-972a-c46767df860f width="200" height="200">


  - model_train.ipynb 파일에는 제안된 모델을 학습시키고 저장한다.
    - 학습된 모델과 train, validation LOSS는 model_save폴더에 저장된다.
    - RB-AE train, validation loss(MSE) with train dataset
    <img src=https://github.com/user-attachments/assets/5ca435f6-d5a5-4a5c-a475-0bb7a4d8d110 width="300" height="150">

  - results.ipynb 파일에선 model_save폴더에 저장된 모델들을 load해 여러 평가지표를 계산하고 시각화한다. 그후 제안될 모델에 Grad-CAM을 적용시킨다.

<img src=https://github.com/user-attachments/assets/656905f0-7816-41f0-b110-dbb62edda92f >
<img src=https://github.com/user-attachments/assets/62b58f4e-7bec-463f-8474-cb183e0cbd09 >
<img src=https://github.com/user-attachments/assets/831c25bb-513e-4889-8b90-457d9ccd838e >

