qm9_folder=./results_reported/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1
zinc250k_folder=./results_reported/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask



time=2
GraphCG_editing_list=(GraphCG_editing_PCA)
embedding_function_list=(PCA_Embedding)
num_direction_list=(10)

num_sample_list=(10)

for embedding_function in "${embedding_function_list[@]}"; do
for GraphCG_editing in "${GraphCG_editing_list[@]}"; do
for num_direction in "${num_direction_list[@]}"; do
for num_sample in "${num_sample_list[@]}"; do

    num_manipulation=200
    data_name=qm9
    model_folder="$qm9_folder"

    output_folder=to_delete/"$data_name"/"$GraphCG_editing"_"$embedding_function"/"$num_direction"_"$num_sample"
    mkdir -p "$output_folder"
    output_file="$output_folder"/output.txt

    if [[ ! -f "$output_file" ]]; then
        python step_02_GraphCG.py \
        --model_dir "$model_folder" \
        -snapshot model_snapshot_epoch_200 --gpu 0 \
        --hyperparams-path moflow-params.json \
        --batch-size 256 \
        --data_name "$data_name"  \
        --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
        --num_direction "$num_direction" \
        --verbose 1 \
        --embedding_function="$embedding_function" --GraphCG_editing="$GraphCG_editing" \
        --output_folder="$output_folder" > "$output_file"
    fi


    num_manipulation=100
    data_name=zinc250k
    model_folder="$zinc250k_folder"
    
    output_folder=to_delete/"$data_name"/"$GraphCG_editing"_"$embedding_function"/"$num_direction"_"$num_sample"
    mkdir -p "$output_folder"
    output_file="$output_folder"/output.txt

    if [[ ! -f "$output_file" ]]; then
        python step_02_GraphCG.py \
        --model_dir "$model_folder" \
        -snapshot model_snapshot_epoch_200 --gpu 0 \
        --hyperparams-path moflow-params.json \
        --batch-size 256 \
        --data_name "$data_name"  \
        --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
        --num_direction "$num_direction" \
        --verbose 1 \
        --embedding_function="$embedding_function" --GraphCG_editing="$GraphCG_editing" \
        --output_folder="$output_folder" > "$output_file"
    fi


done
done
done
done

