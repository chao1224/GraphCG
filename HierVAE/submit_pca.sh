
time=2
GraphCG_editing_list=(GraphCG_editing_PCA)
embedding_function_list=(PCA_Embedding)
num_direction_list=(10)




time=2
GraphCG_editing_list=(GraphCG_editing_PCA)
embedding_function_list=(PCA_Embedding)
num_direction_list=(2 3)


for embedding_function in "${embedding_function_list[@]}"; do
for GraphCG_editing in "${GraphCG_editing_list[@]}"; do
for num_direction in "${num_direction_list[@]}"; do
    
    epochs=100
    num_manipulation=200
    num_sample=100
    data_name=qm9
    model=results/qm9/model.ckpt

    output_folder=results_manipulation/"$data_name"/"$GraphCG_editing"_"$embedding_function"/"$num_direction"
    mkdir -p "$output_folder"
    output_file="$output_folder"/output.txt

    if [[ ! -f "$output_file" ]]; then
        sbatch --gres=gpu:1 -c 8 --mem=32G -t "$time":59:00  --account=xxx --qos=high --job-name=hie_pca \
        --output="$output_file" \
        ./run_step_02_GraphCG.sh \
        --model "$model" \
        --train hgraph2graph/data/"$data_name"/all.txt \
        --vocab hgraph2graph/data/"$data_name"/vocab.txt \
        --epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
        --num_direction "$num_direction" \
        --embedding_function="$embedding_function" --GraphCG_editing="$GraphCG_editing" \
        --output_folder="$output_folder"
    fi




    
    epochs=100
    num_manipulation=200
    num_sample=50
    data_name=chembl
    model=hgraph2graph/ckpt/chembl-pretrained/model.ckpt
    
    output_folder=results_manipulation/"$data_name"/"$GraphCG_editing"_"$embedding_function"/"$num_direction"
    mkdir -p "$output_folder"
    output_file="$output_folder"/output.txt

    if [[ ! -f "$output_file" ]]; then
        sbatch --gres=gpu:1 -c 8 --mem=32G -t "$time":59:00  --account=xxx --qos=high --job-name=hie_pca \
        --output="$output_file" \
        ./run_step_02_GraphCG.sh \
        --model "$model" \
        --train hgraph2graph/data/"$data_name"/all.txt \
        --vocab hgraph2graph/data/"$data_name"/vocab.txt \
        --epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
        --num_direction "$num_direction" \
        --embedding_function="$embedding_function" --GraphCG_editing="$GraphCG_editing" \
        --output_folder="$output_folder"
    fi

done
done
done
