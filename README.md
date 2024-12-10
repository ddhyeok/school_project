
* 데이터 출처 : <https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71688>

front 구강 이미지 데이터


### school_project(파이썬 기반 딥러닝 코드 제출)



# git-구강이미지
  
  - Grad-CAM.py 파일에는 Grad-CAM이 구현되어있다.
    
  - method.py 파일에는 모델 학습, 검증, 시각화에 필요한 함수가 정의 되어있다.

    - Method
        
      - train                   :분류모델 train
      - validate                :분류모델 validation
      - test                    :분류모델 test
      - model_train_eval        :분류모델 train, validation 후 test
      - CAE_train               :CAE train
      - CAE_ validate           :CAE validation
      - CAE_train_eval          :CAE train, validation
      - plot_loss_and_accuracy  :train, validation loss와accuracy plot+혼동행렬 plot
      - test_for_inference_time :inference time 측정을 위한 test
    
  
  
  - models 폴더에는 본 프로젝트에서 제안한 모델(CAE_v2, Classfier_v2)이 구현되어있다.
  <img src=https://github.com/user-attachments/assets/501208f3-2bbb-46a0-903e-b86be240737f width="500" height="300">

    - CAE_v2        :RB-AE구현
    - Classfier_v2  :RB-AE를 입력받아 ResFC 구현
  
  - Data_preprocess_split.ipynb 파일에서는 원천데이터를 전처리하고 split하여 저장한다.
  
    - 전처리 Method
      - load_images_from_folder :image를 다운 샘플링 후 0 패딩 추가
      - load_label_from_folder  :json파일로 부터 충치 여부 라벨 추출
      - save_images_with_names  :폴더로 저장
      - create_dataset          :전처리된 이미지를 split하여 output_dataset 생성
  
    - Data/image_train_front, Data/label_train_front 폴더에 들어있는 원천데이터를 전처리한다.
    
    - Data/image_preprocessed_0, Data/image_preprocessed_1 폴더에 전처리된 이미지를 저정한다.
    
    - data를 split 하여 output_dataset 폴더 안에 저장한다. (각 데이터 셋의 클래스 비는 73.1(음성):26.9(양성)로 동일하다)
      
    - output_dataset

      ├── train ── 0,1

      ├── test ── 0,1

      └── val ── 0,1
    <img src=https://github.com/user-attachments/assets/729dd97f-de1d-44a8-a793-981f63f3a7f9 width="200" height="100">
    
    <figcaption>예시 구강이미지</figcaption>
    
    <img src=https://github.com/user-attachments/assets/5488c939-63ce-4f03-805d-65b81e8350ee width="200" height="200">
    <img src=https://github.com/user-attachments/assets/c3e1283b-18c0-4a74-972a-c46767df860f width="200" height="200">


  - model_train.ipynb 파일에는 제안된 모델을 학습시키고 저장한다.
    - 학습된 모델과 train, validation LOSS는 model_save폴더에 저장된다.
    - RB-AE train, validation loss(MSE) with train dataset
    <img src=https://github.com/user-attachments/assets/5ca435f6-d5a5-4a5c-a475-0bb7a4d8d110 width="300" height="150">

  - results.ipynb 파일에선 model_save폴더에 비교군 모델들과 제안할 모델을 load해 여러 평가지표를 계산하고 시각화한다. 그후 제안될 모델에 Grad-CAM을 적용시켜 시각화한다.
<img src=https://github.com/user-attachments/assets/62b58f4e-7bec-463f-8474-cb183e0cbd09 width="1000" height="100">
<img src=https://github.com/user-attachments/assets/656905f0-7816-41f0-b110-dbb62edda92f width="400" height="200">
<img src=https://github.com/user-attachments/assets/831c25bb-513e-4889-8b90-457d9ccd838e width="300" height="200">
<img src=https://github.com/user-attachments/assets/cfe2ba72-a5d1-4fbb-80c7-3a1fce0212eb width="700" height="400">


# sever

  - Flask를 통한 간단한 웹서비스 구현
  - models 사용할 모델 구현 파일
    
  - static
    - img        : HTML에서 사용할 이미지 저장
    - input.png  : 입력받은 이미지
    - reulst.png : Grad-CAM overlay한 이미지
 
  - templates

<img src=https://github.com/user-attachments/assets/e418bf1a-ef84-4cf1-8528-03c0463f593d width="700" height="400">
  
    1. index.html  : 홈페이지

        - 검사할 구강 이미지를 input
        - "검사하기" 클릭

<img src=https://github.com/user-attachments/assets/a8b7fe4c-b585-46f8-9d4c-c1d836f4a6cd width="700" height="400">
 
    2. results.html: result 페이지
   
        - 안내문과, 전처리된 input이미지를 보여줌
        - 이미지에 마우스를 "호버링"하면 Grad-CAM을 overlay하여 보여줌

  - app.py
    - flask 라이브러리로 index.html, result.html 을 라우팅함
    - index.html에서 image를 입력받고 검사하기를 누르면
    - 전처리, 모델 입력 후 출력을 result.html와 보여줌
   
      
