### school_project(파이썬 기반 딥러닝 코드 제출)

# git-구강이미지
  
  -method.py 파일에는 모델 학습, 검증, 시각화에 필요한 함수가 정의 되어있다
  
  -Grad-CAM.py 파일에는 Grad-CAM이 구현되어있다.
  
  -models 폴더에는 본 프로젝트에서 제안한 모델(CAE_v2, Classfier_v2)이 구현되어있다
  
  -Data_preprocess_split.ipynb 파일에서는 원천데이터를 전처리하고 split하여 저장한다.
  
    -Data/image_train_front,Data/label_train_front 폴더에 들어있는 원천데이터를 전처리한다.
    
    -Data/image_preprocessed_0,Data/image_preprocessed_1 폴더에 전처리된 이미지를 저정한다.
    
    -data를 split 하여 output_dataset 폴더 안에 저장한다. output_dataset
                                                        ├── train ── 0,1
                                                        ├── test ── 0,1
                                                        └── val ── 0,1
    -![image](https://github.com/user-attachments/assets/5488c939-63ce-4f03-805d-65b81e8350ee)

    -![image](https://github.com/user-attachments/assets/c3e1283b-18c0-4a74-972a-c46767df860f)

  -model_train.ipynb 파일에는 제안된 모델을 학습시키고 저장한다.
    -학습된 모델과 train, validation LOSS는 model_save폴더에 저장된다.


  -results.ipynb 파일에선 model_save폴더에 저장된 모델들을 load해 여러 평가지표를 계산하고 시각화한다. 그후 제안될 모델에 Grad-CAM을 적용시킨다.


