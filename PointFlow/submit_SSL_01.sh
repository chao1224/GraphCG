epochs=200
num_manipulation=10
num_sample=500
time=2

cates_list=(airplane car chair)
contrastive_SSL_list=(contrastive_SSL_01)
embedding_function_list=(Direction_Embedding_01 Direction_Embedding_02 Direction_Embedding_03)
num_direction_list=(10)
SSL_noise_level_list=(0.1 1 2)

alpha_step_option_list=(random first_last)

alpha_01_list=(1 2)
alpha_02_list=(0 1)
alpha_03_list=(0)

for cates in "${cates_list[@]}"; do
for contrastive_SSL in "${contrastive_SSL_list[@]}"; do
for embedding_function in "${embedding_function_list[@]}"; do
for num_direction in "${num_direction_list[@]}"; do
for SSL_noise_level in "${SSL_noise_level_list[@]}"; do

for alpha_01 in "${alpha_01_list[@]}"; do
for alpha_02 in "${alpha_02_list[@]}"; do
for alpha_03 in "${alpha_03_list[@]}"; do

for alpha_step_option in "${alpha_step_option_list[@]}"; do
    if [ "$alpha_step_option" = "random" ]; then
        alpha_step_option_random_num_list=(1)
    else
        alpha_step_option_random_num_list=( $( expr 2 '*' "$num_direction" ))
    fi
for alpha_step_option_random_num in "${alpha_step_option_random_num_list[@]}"; do

        output_folder=results_manipulation/"$cates"/"$contrastive_SSL"_"$embedding_function"/"$num_direction"_"$alpha_step_option"_"$alpha_step_option_random_num"_"$alpha_01"_"$alpha_02"_"$alpha_03"_"$SSL_noise_level"
        mkdir -p "$output_folder"
        output_file="$output_folder"/output.txt

        echo "$output_file"

        if [[ ! -f "$output_file" ]]; then
            sbatch --gres=gpu:1 -c 8 --mem=32G -t "$time":59:00  --account="$account" --qos=high --job-name=01 \
            --output="$output_file" \
            ./run_step_02_GraphCG.sh \
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
        fi

done
done
done
done
done
done
done
done
done
done
