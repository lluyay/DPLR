# train dynamic_pseudo_label_refinement on ACDC
python train_dynamic_pseudo_label_refinement_2D.py --root_path ../data/ACDC --exp ACDC/Dynamic_Pseudo_Label_Refinement --num_classes 4 --labeled_num 3 --thr 0.43 0.58 &&

python train_dynamic_pseudo_label_refinement_2D.py --root_path ../data/ACDC --exp ACDC/Dynamic_Pseudo_Label_Refinement --num_classes 4 --labeled_num 7 --thr 0.48 0.53

# train dynamic_pseudo_label_refinement on BraTS
python train_dynamic_pseudo_label_refinement_3D.py --root_path ../data/BraTS2019 --exp BraTS2019/Dynamic_Pseudo_Label_Refinement --labeled_num 25 && \

python train_dynamic_pseudo_label_refinement_3D.py --root_path ../data/BraTS2019 --exp BraTS2019/Dynamic_Pseudo_Label_Refinement --labeled_num 50
