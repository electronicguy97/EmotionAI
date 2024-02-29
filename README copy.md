![middle](https://capsule-render.vercel.app/api?type=cylinder&color=0147FF&height=150&section=header&text=Wassup&fontColor=FFFFFF&fontSize=70&animation=fadeIn&fontAlignY=55)
### Language
[한국어 Readme](https://github.com/electronicguy97/est_wassup_03/blob/main/exam/korean/korean.md)

### Project
We used 500,000 Asian amounts.<br>
A total of 7 emotions were used: happy, angry, anxious, peaceful, pain, and sad.<br>
We used YOLOv8-face as a preprocessing method to cut out only the face to save manpower and learning time.<br>
Repvgg, VIT (Vision Transformer), and YOLO were used as 2-stage models, and YOLO was used as 1-stage model.<br>
Streamlit adds various new features.<br>
![image](https://github.com/electronicguy97/est_wassup_03/assets/103613730/41417652-dea9-4123-a3d9-5332af6f4bc6)



### Use
GPU server : 4GPU A-100 (AWS)
OS : Linux
Language : Python

### Team
- [DoYeon Kim](https://github.com/electronicguy97) - Team Leader
- [HyunJun Kang](https://github.com/)
- [Jongseong Kim](https://github.com/JamieSKinard)
- [Chaewook Lee](https://github.com/leecw12)
- [HaNeul Pyeon](https://github.com/Haneul1002)

### Experiment Report
|Experiment Report|Presentation Materials|
|---|---|
|[결과보고서.pdf](https://github.com/electronicguy97/est_wassup_03/files/14441069/default.pdf)|[감정AI발표자료.pdf](https://github.com/electronicguy97/est_wassup_03/files/14441162/AI.pdf)|
|[결과보고서.docx](https://github.com/electronicguy97/est_wassup_03/files/14441072/default.docx)||

### How to Install
```bash
git clone https://github.com/JamieSKinard/est_wassup_03.git
cd est_wassup_03
pip install -r requirements.txt
pip install -e .
```
or
```bash
# using conda
conda env create -f env.yaml
```

### How to Pretreatment
```bash
# check default path
python preprocess/preprocess.py --data-dir {your_data_path}
```
Create a cropped photo after face recognition with the yolo8n-face model

### How to Learn(2Stage Model)
```bash
python main.py --data-dir {your_data_path}
### Check defult
### Change model -> choice(defalut = RepVGG, VIT)
python main.py -mn VIT --data-dir {your_data_path} -mp {your_model} -mn {Reppvgg or VIT}
```
History and model aved in Models folder/{your_choice_model}

### How to Learn(1Stage Model)
Preprocessing is possible with box_labeling_yolov8.ipynb in the folder called preprocess.
and Go to the file named YOLO.ipynb and Just Shift + F5

### How to evaluation
```bash
python eval.py --data-dir {your_test_folder_path} -mp {your_model_path} -mn {Repvgg, VIT}
```
We used f1, R2, Precision, and recall as metrics.

### result

![image](https://github.com/electronicguy97/est_wassup_03/assets/103613730/ae6e255c-3d7f-4323-8486-b8ed2c56094d)

||YOLO(1Stage)|YOLO(2Stage)|ReppVgg|VIT|
|---|---|---|---|---|
|val_loss|0.233|0.533|1.470|1.251|
|train_acc|||78.3%|72.2%|
|val_acc||73.9%|68.7%|62.5%|

### Example(Mask)

![image](https://github.com/JamieSKinard/est_wassup_03/assets/103613730/028820b2-4d97-4a71-a405-96affe3465eb)
![image](https://github.com/JamieSKinard/est_wassup_03/assets/103613730/3a7145fb-6e0c-4cc2-880c-9da0bf5c71f9)


<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src = "https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<a href="https://code.visualstudio.com/" onClick=""><img src="https://img.shields.io/badge/VSC-007ACC?style=flat-square&logo=Visual Studio Code&logoColor=white"/></a>
<a href="https://www.linux.org/" onClick=""><img src="https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=Linux&logoColor=white"/></a>
<a href="https://git-scm.com/" onClick=""><img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=Git&logoColor=white"/></a>
