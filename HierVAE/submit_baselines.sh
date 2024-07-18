time=5

contrastive_SSL_list=(
    contrastive_SSL_random
    contrastive_SSL_variance_high
)
embedding_function="none"
num_direction_list=(10)
num_sample_list=(100 500)


for contrastive_SSL in "${contrastive_SSL_list[@]}"; do
for num_direction in "${num_direction_list[@]}"; do
for num_sample in "${num_sample_list[@]}"; do
 
    epochs=100
    num_manipulation=200
    data_name=qm9
    model=results/qm9/model.ckpt

    output_folder=results_manipulation/"$data_name"/"$contrastive_SSL"/"$num_direction"_"$num_sample"
    mkdir -p "$output_folder"
    output_file="$output_folder"/output.txt

    if [[ ! -f "$output_file" ]]; then
        sbatch --gres=gpu:1 -c 8 --mem=32G -t "$time":59:00  --account=xxx --qos=high --job-name=hie_base \
        --output="$output_file" \
        ./run_step_02_GraphCG.sh \
        --model "$model" \
        --train hgraph2graph/data/"$data_name"/all.txt \
        --vocab hgraph2graph/data/"$data_name"/vocab.txt \
        --epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
        --num_direction "$num_direction" \
        --embedding_function="$embedding_function" --contrastive_SSL="$contrastive_SSL" \
        --alpha_step_option=first_last \
        --alpha_01=1 --alpha_02=0 --alpha_03=0 \
        --output_folder="$output_folder"
    fi

    
    epochs=100
    num_manipulation=100
    data_name=chembl
    model=hgraph2graph/ckpt/chembl-pretrained/model.ckpt
    
    output_folder=results_manipulation/"$data_name"/"$contrastive_SSL"/"$num_direction"_"$num_sample"
    mkdir -p "$output_folder"
    output_file="$output_folder"/output.txt

    if [[ ! -f "$output_file" ]]; then
        sbatch --gres=gpu:1 -c 8 --mem=32G -t "$time":59:00  --account=xxx --qos=high --job-name=hie_base \
        --output="$output_file" \
        ./run_step_02_GraphCG.sh \
        --model "$model" \
        --train hgraph2graph/data/"$data_name"/all.txt \
        --vocab hgraph2graph/data/"$data_name"/vocab.txt \
        --epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
        --num_direction "$num_direction" \
        --embedding_function="$embedding_function" --contrastive_SSL="$contrastive_SSL" \
        --alpha_step_option=first_last \
        --alpha_01=1 --alpha_02=0 --alpha_03=0 \
        --output_folder="$output_folder"
    fi

done
done
done
