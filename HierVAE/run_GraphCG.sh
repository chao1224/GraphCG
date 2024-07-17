
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
--contrastive_SSL=contrastive_SSL_random \
--verbose=1


######### baseline variance high ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--contrastive_SSL=contrastive_SSL_variance_high \
--verbose=1


######### baseline variance low ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--contrastive_SSL=contrastive_SSL_variance_low \
--verbose=1


######### baseline PCA ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--contrastive_SSL=contrastive_SSL_PCA \
--verbose=1


######### baseline umap ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--contrastive_SSL=contrastive_SSL_UMAP \
--verbose=1


# ######### baseline DisCo ##########
# python step_02_GraphCG.py \
# --model "$model" \
# --train hgraph2graph/data/"$data_name"/all.txt \
# --vocab hgraph2graph/data/"$data_name"/vocab.txt \
# --epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
# --num_direction "$num_direction" \
# --contrastive_SSL=contrastive_SSL_DisCo --embedding_function=MoFlowDisCo \
# --alpha_step_option=first_last \
# --alpha_01=1 --alpha_02=0 --alpha_03=0 \
# --verbose=1








epochs=100
num_manipulation=100
num_sample=500
num_direction=10

data_name=chembl
model=hgraph2graph/ckpt/chembl-pretrained/model.ckpt
######### chembl ##########
######### baseline one-hot ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--embedding_function="$embedding_function" --contrastive_SSL=contrastive_SSL_onehot \
--verbose=1


######### baseline variance high ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--embedding_function="$embedding_function" --contrastive_SSL=contrastive_SSL_variance_high \
--verbose=1


######### baseline variance low ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
--num_direction "$num_direction" \
--embedding_function="$embedding_function" --contrastive_SSL=contrastive_SSL_variance_low \
--verbose=1


######### baseline umap ##########
python step_02_GraphCG.py \
--model "$model" \
--train hgraph2graph/data/"$data_name"/all.txt \
--vocab hgraph2graph/data/"$data_name"/vocab.txt \
--epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample 100 \
--num_direction 3 \
--embedding_function="$embedding_function" --contrastive_SSL=contrastive_SSL_UMAP \
--verbose=1


# ######### baseline DisCo ##########
# python step_02_GraphCG.py \
# --model "$model" \
# --train hgraph2graph/data/"$data_name"/all.txt \
# --vocab hgraph2graph/data/"$data_name"/vocab.txt \
# --epochs "$epochs" --num_manipulation "$num_manipulation" --num_sample "$num_sample" \
# --num_direction "$num_direction" \
# --contrastive_SSL=contrastive_SSL_DisCo --embedding_function=MoFlowDisCo \
# --alpha_step_option=first_last \
# --alpha_01=1 --alpha_02=0 --alpha_03=0 \
# --verbose=1
