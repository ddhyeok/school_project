### school_project(파이썬 기반 딥러닝 코드 제출)

# git-구강이미지
  -method.py 파일에는 모델 학습, 검증, 시각화에 필요한 함수가 정의 되어있다
  -Grad-CAM.py 파일에는 Grad-CAM이 구현되어있다.
  -models 폴더에는 본 프로젝트에서 제안한 모델(CAE_v2, Classfier_v2)이 구현되어있다

-Data_preprocess_split.ipynb파일에서는 
Data/image_train_front
Data/label_train_front
에 들어있는 원천데이터를 전처리하여
Data/image_preprocessed_0
Data/image_preprocessed_1 
로 저정한다.

그 후 
output_dataset에 포함된 train, test, val에 포함된 0,1파일로 이미지를 split한다. 

-model_train.ipynb 파일에는 제안된 모델을 학습시키고 저장한다.
저장된 모델의 train, validation 과정은 model_save폴더에 저장된다.

-results.ipynb 파일에선 model_save폴더에 저장된 모델들을 load해 여러 평가지표를 계산하고 시각화한다.
그후 제안될 모델에 Grad-CAM을 적용시킨다.


