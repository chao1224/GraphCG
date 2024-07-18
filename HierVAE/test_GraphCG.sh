
epochs=100
num_manipulation=200
num_sample=500
num_direction=10



epochs=5
num_manipulation=3
num_sample=5
num_direction=5


data_name=qm9
model=results_reported/qm9/model.ckpt
######### qm9 ##########
######### baseline random ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--GraphCG_editing=GraphCG_editing_random \
--verbose=1


######### baseline variance high ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--GraphCG_editing=GraphCG_editing_variance_high \
--verbose=1


######### baseline PCA ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--GraphCG_editing=GraphCG_editing_PCA \
--verbose=1


######### GraphCG ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--embedding_function=Direction_Embedding_02 \
--verbose=1





epochs=100
num_manipulation=100
num_sample=500
num_direction=10

data_name=chembl
model=hgraph2graph/ckpt/chembl-pretrained/model.ckpt
######### chembl ##########
######### baseline random ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--GraphCG_editing=GraphCG_editing_random \
--verbose=1


######### baseline variance high ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--embedding_function="$embedding_function" --GraphCG_editing=GraphCG_editing_variance_high \
--verbose=1


######### GraphCG ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--embedding_function=Direction_Embedding_02 \
--verbose=1