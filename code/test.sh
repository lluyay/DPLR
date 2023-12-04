# test dynamic_pseudo_label_refinement on ACDC
python test_2D.py --root_path ../data/ACDC --exp ACDC/Dynamic_Pseudo_Label_Refinement --num_classes 4 --labeled_num 3 &&
python test_2D.py --root_path ../data/ACDC --exp ACDC/Dynamic_Pseudo_Label_Refinement --num_classes 4 --labeled_num 7

# test dynamic_pseudo_label_refinement on BraTS
python test_3D.py --root_path ../data/BraTS2019 --exp BraTS2019/Dynamic_Pseudo_Label_Refinement --labeled_num 25 &&
python test_3D.py --root_path ../data/BraTS2019 --exp BraTS2019/Dynamic_Pseudo_Label_Refinement --labeled_num 50