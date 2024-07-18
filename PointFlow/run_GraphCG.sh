epochs=10
num_manipulation=10
num_sample=100


cates=airplane
contrastive_SSL=contrastive_SSL_01
embedding_function=Direction_Embedding_01
num_direction=3
SSL_noise_level=0.1

alpha_step_option=random
alpha_01=1
alpha_02=1
alpha_03=0
alpha_step_option_random_num=1


python step_02_GraphCG.py \
--dataset_type shapenet15k \
--cates "$cates" \
--num_sample_points 8192 \
--dims 512-512-512 \
--latent_dims 256-256 \
--use_latent_flow \
--resume_checkpoint PointFlow/pretrained_models/gen/"$cates"/checkpoint.pt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--contrastive_SSL="$contrastive_SSL" --embedding_function="$embedding_function" \
--alpha_step_option="$alpha_step_option" --alpha_step_option_random_num="$alpha_step_option_random_num" \
--SSL_noise_level="$SSL_noise_level" \
--alpha_01="$alpha_01" --alpha_02="$alpha_02" --alpha_03="$alpha_03" \
--output_folder="$output_folder" \
--data_dir "ShapeNetCore.v2.PC15k"