qm9_folder=./results_reported/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1
zinc250k_folder=./results_reported/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask




time=11
contrastive_SSL_list=(
    contrastive_SSL_random
    contrastive_SSL_variance_high
    contrastive_SSL_DisCo
)
num_direction_list=(10)
num_sample_list=(100 500)

for contrastive_SSL in "${contrastive_SSL_list[@]}"; do
if [ "$contrastive_SSL" = "contrastive_SSL_DisCo" ]; then
    embedding_function="MoFlowDisCo"
else
    embedding_function="none"
fi
for num_direction in "${num_direction_list[@]}"; do
for num_sample in "${num_sample_list[@]}"; do
    
    epochs=200
    num_manipulation=200
    num_sample=500
    data_name=qm9
    model_folder="$qm9_folder"

    output_folder=results_manipulation/"$data_name"/"$contrastive_SSL"_"$embedding_function"/"$num_direction"_"$num_sample"
    mkdir -p "$output_folder"
    output_file="$output_folder"/output.txt

    if [[ ! -f "$output_file" ]]; then
        sbatch --gres=gpu:1 -c 8 --mem=32G -t "$time":59:00  --account=xxx --qos=high --job-name=mo_baselines \
        --output="$output_file" \
        ./run_step_02_GraphCG.sh \
        --model_dir "$model_folder" \
        -snapshot model_snapshot_epoch_200 --gpu 0 \
        --hyperparams-path moflow-params.json \
        --batch-size 256 \
        --data_name "$data_name"  \
        --epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
        --num_direction "$num_direction" \
        --embedding_function="$embedding_function" --contrastive_SSL="$contrastive_SSL" \
        --output_folder="$output_folder"
    fi




    epochs=200
    num_manipulation=100
    data_name=zinc250k
    model_folder="$zinc250k_folder"
    
    output_folder=results_manipulation/"$data_name"/"$contrastive_SSL"/"$num_direction"_"$num_sample"
    output_folder=results_manipulation/"$data_name"/"$contrastive_SSL"/"$num_direction"_"$num_sample"_"$epochs"
    mkdir -p "$output_folder"
    output_file="$output_folder"/output.txt

    if [[ ! -f "$output_file" ]]; then
        sbatch --gres=gpu:1 -c 8 --mem=32G -t "$time":59:00  --account=xxx --qos=high --job-name=mo_baselines \
        --output="$output_file" \
        ./run_step_02_GraphCG.sh \
        --model_dir "$model_folder" \
        -snapshot model_snapshot_epoch_200 --gpu 0 \
        --hyperparams-path moflow-params.json \
        --batch-size 256 \
        --data_name "$data_name"  \
        --epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
        --num_direction "$num_direction" \
        --embedding_function="$embedding_function" --contrastive_SSL="$contrastive_SSL" \
        --alpha_step_option=first_last \
        --output_folder="$output_folder"
    fi

done
done
done
