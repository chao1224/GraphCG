qm9_folder=./results_reported/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1
zinc250k_folder=./results_reported/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask




time=2
embedding_function_list=(Direction_Embedding_01 Direction_Embedding_02 Direction_Embedding_03)
GraphCG_editing_list=(GraphCG_editing_02)
num_direction_list=(10 32)

alpha_step_option_list=(random first_last)

alpha_01_list=(1 2)
alpha_02_list=(0 1)
alpha_03_list=(0 1)
num_sample_list=(100 500)



for embedding_function in "${embedding_function_list[@]}"; do
for GraphCG_editing in "${GraphCG_editing_list[@]}"; do
for num_direction in "${num_direction_list[@]}"; do

for alpha_01 in "${alpha_01_list[@]}"; do
for alpha_02 in "${alpha_02_list[@]}"; do
for alpha_03 in "${alpha_03_list[@]}"; do
for num_sample in "${num_sample_list[@]}"; do

for alpha_step_option in "${alpha_step_option_list[@]}"; do
    if [ "$alpha_step_option" = "random" ]; then
        alpha_step_option_random_num_list=(1)
    else
        alpha_step_option_random_num_list=( $( expr 2 '*' "$num_direction" ))
    fi
for alpha_step_option_random_num in "${alpha_step_option_random_num_list[@]}"; do

    epochs=200
    num_manipulation=200
    data_name=qm9
    model_folder="$qm9_folder"

    output_folder=results_manipulation/"$data_name"/"$GraphCG_editing"_"$embedding_function"/"$num_direction"_"$num_sample"_"$alpha_step_option"_"$alpha_step_option_random_num"_"$alpha_01"_"$alpha_02"_"$alpha_03"
    mkdir -p "$output_folder"
    output_file="$output_folder"/output.txt

    if [[ ! -f "$output_file" ]]; then
        sbatch --gres=gpu:1 -c 8 --mem=32G -t "$time":59:00  --account=xxx --qos=high --job-name=mo_02 \
        --output="$output_file" \
        ./run_step_02_GraphCG.sh \
        --model_dir "$model_folder" \
        -snapshot model_snapshot_epoch_200 --gpu 0 \
        --hyperparams-path moflow-params.json \
        --batch-size 256 \
        --data_name "$data_name"  \
        --epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
        --num_direction "$num_direction" \
        --embedding_function="$embedding_function" --GraphCG_editing="$GraphCG_editing" \
        --alpha_step_option="$alpha_step_option" --alpha_step_option_random_num="$alpha_step_option_random_num" \
        --alpha_01="$alpha_01" --alpha_02="$alpha_02" --alpha_03="$alpha_03" \
        --output_folder="$output_folder"
    fi




    epochs=100
    num_manipulation=100
    data_name=zinc250k
    model_folder="$zinc250k_folder"
    
    output_folder=results_manipulation/"$data_name"/"$GraphCG_editing"_"$embedding_function"/"$num_direction"_"$num_sample"_"$alpha_step_option"_"$alpha_step_option_random_num"_"$alpha_01"_"$alpha_02"_"$alpha_03"_"$epochs"

    mkdir -p "$output_folder"
    output_file="$output_folder"/output.txt

    if [[ ! -f "$output_file" ]]; then
        echo "undone"
        sbatch --gres=gpu:1 -c 8 --mem=32G -t "$time":59:00  --account=xxx --qos=high --job-name=mo_02 \
        --output="$output_file" \
        ./run_step_02_GraphCG.sh \
        --model_dir "$model_folder" \
        -snapshot model_snapshot_epoch_200 --gpu 0 \
        --hyperparams-path moflow-params.json \
        --batch-size 256 \
        --data_name "$data_name"  \
        --epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
        --num_direction "$num_direction" \
        --embedding_function="$embedding_function" --GraphCG_editing="$GraphCG_editing" \
        --alpha_step_option="$alpha_step_option" --alpha_step_option_random_num="$alpha_step_option_random_num" \
        --alpha_01="$alpha_01" --alpha_02="$alpha_02" --alpha_03="$alpha_03" \
        --output_folder="$output_folder"
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
