qm9_folder=./results_reported/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1
zinc250k_folder=./results_reported/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask
chembl_folder=./results_reported/chembl




######### qm9 ##########
######### baseline random ##########
python step_02_GraphCG.py --model_dir "$qm9_folder" \
-snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9  --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 10 --num_direction 10 --num_manipulation 10 --num_sample=100 \
--GraphCG_editing=GraphCG_editing_random \
--verbose=1

######### baseline variance high ##########
python step_02_GraphCG.py --model_dir "$qm9_folder" \
-snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9  --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 100 --num_direction 10 --num_manipulation 100 --num_sample=100 \
--GraphCG_editing=GraphCG_editing_variance_high \
--verbose=1

######### baseline PCA ##########
python step_02_GraphCG.py --model_dir "$qm9_folder" \
-snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9  --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 100 --num_direction 10 --num_manipulation 100 --num_sample=100 \
--GraphCG_editing=GraphCG_editing_PCA \
--verbose=1

######### baseline DisCo ##########
python step_02_GraphCG.py --model_dir "$qm9_folder" \
-snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9  --hyperparams-path moflow-params.json \
--epochs 50 --num_direction 10 --num_manipulation 100 --num_sample=100 \
--batch-size 10 \
--GraphCG_editing=GraphCG_editing_DisCo --embedding_function=MoFlowDisCo \
--alpha_step_option=first_last \
--alpha_01=1 --alpha_02=0 --alpha_03=0 \
--verbose=1




######### GraphCG ##########
python step_02_GraphCG.py --model_dir "$qm9_folder" \
-snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9  --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 10 --num_direction 2 --num_manipulation 100 --num_sample=100 \
--embedding_function=Direction_Embedding_02 \
--verbose=1


python step_02_GraphCG.py --model_dir "$qm9_folder" \
-snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9  --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 100 --num_direction 10 --num_manipulation 100 --num_sample=100 \
--embedding_function=Direction_Embedding_01 --alpha_03=1 \
--verbose=1


python step_02_GraphCG.py --model_dir "$qm9_folder" \
-snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9  --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 100 --num_direction 10 --num_manipulation 100 --num_sample=100 \
--embedding_function=Direction_Embedding_03 --alpha_03=1 \
--verbose=1



python step_02_GraphCG.py --model_dir "$qm9_folder" \
-snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9  --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 20 --num_direction 10 --num_manipulation 100 --num_sample 15 \
--embedding_function=Direction_Embedding_03 \
--GraphCG_editing=GraphCG_editing_02 \
--verbose=1





python step_02_GraphCG.py --model_dir "$qm9_folder" \
-snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9  --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 20 --num_direction 10 --num_manipulation 1000 --num_sample 15 \
--embedding_function=Direction_Embedding_03 \
--GraphCG_editing=GraphCG_editing_02 \
--verbose=1





######### zinc250k ##########
######### baseline random ##########
python step_02_GraphCG.py --model_dir "$zinc250k_folder" \
-snapshot model_snapshot_epoch_200 --gpu 0 --data_name zinc250k  --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 100 --num_direction 10 --num_manipulation 100 --num_sample=100 \
--GraphCG_editing=GraphCG_editing_random \
--verbose=1

######### baseline variance high ##########
python step_02_GraphCG.py --model_dir "$zinc250k_folder" \
-snapshot model_snapshot_epoch_200 --gpu 0 --data_name zinc250k  --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 100 --num_direction 10 --num_manipulation 100 --num_sample=100 \
--GraphCG_editing=GraphCG_editing_variance_high \
--verbose=1

######### PCA ##########
python step_02_GraphCG.py --model_dir "$zinc250k_folder" \
-snapshot model_snapshot_epoch_200 --gpu 0 --data_name zinc250k  --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 100 --num_direction 3 --num_manipulation 100 --num_sample=100 \
--GraphCG_editing=GraphCG_editing_PCA \
--verbose=1

######### baseline DisCo ##########
python step_02_GraphCG.py --model_dir "$zinc250k_folder" \
-snapshot model_snapshot_epoch_200 --gpu 0 --data_name zinc250k  --hyperparams-path moflow-params.json \
--epochs 10 --num_direction 5 --num_manipulation 100 --num_sample=5 \
--batch-size 10 \
--GraphCG_editing=GraphCG_editing_DisCo --embedding_function=MoFlowDisCo \
--alpha_step_option=first_last \
--alpha_01=1 --alpha_02=0 --alpha_03=0 \
--verbose=1




######### GraphCG ##########
python step_02_GraphCG.py --model_dir "$zinc250k_folder" \
-snapshot model_snapshot_epoch_200 --gpu  0  --data_name zinc250k --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 100 --num_direction 5 --num_manipulation 100 --num_sample=100 \
--embedding_function=Direction_Embedding_01 \
--verbose=1 --lr=1e-2



python step_02_GraphCG.py --model_dir "$zinc250k_folder" \
-snapshot model_snapshot_epoch_200 --gpu  0  --data_name zinc250k --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 100 --num_direction 10 --num_manipulation 100 --num_sample=100 \
--embedding_function=Direction_Embedding_02 \
--verbose=1


python step_02_GraphCG.py --model_dir "$zinc250k_folder" \
-snapshot model_snapshot_epoch_200 --gpu  0  --data_name zinc250k --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 100 --num_direction 10 --num_manipulation 100 --num_sample=100 \
--embedding_function=Direction_Embedding_01 \
--verbose=1



python step_02_GraphCG.py --model_dir "$zinc250k_folder" \
-snapshot model_snapshot_epoch_200 --gpu  0  --data_name zinc250k --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 200 --num_direction 10 --num_manipulation 100 --num_sample=100 \
--GraphCG_editing=GraphCG_editing_02 --embedding_function=Direction_Embedding_03 \
--verbose=1



python step_02_GraphCG.py --model_dir "$zinc250k_folder" \
-snapshot model_snapshot_epoch_200 --gpu  0  --data_name zinc250k --hyperparams-path moflow-params.json \
--batch-size 256 --epochs 20 --num_direction 10 --num_manipulation 200 --num_sample=100 \
--GraphCG_editing=GraphCG_editing_02 --embedding_function=Direction_Embedding_03 \
--verbose=1



# optimal
python step_02_GraphCG.py \
--model_dir ./results_reported/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask \
-snapshot model_snapshot_epoch_200 --gpu 0 --hyperparams-path moflow-params.json \
--batch-size 256 --data_name zinc250k --epochs 20 --num_manipulation 100 \
--num_sample 500 --num_direction 32 --embedding_function=Direction_Embedding_03 \
--GraphCG_editing=GraphCG_editing_01 --alpha_step_option=first_last \
--alpha_step_option_random_num=64 --SSL_noise_level=0.1 --alpha_01=2 --alpha_02=1 --alpha_03=0
#--output_folder=results_manipulation/zinc250k/GraphCG_editing_01_Direction_Embedding_03/32_first_last_64_2_1_0_0.1
