#! /bin/bash



OUTPUT_FILE=/home/lily/wz336/graph2text/test_large_pred
#OUTPUT_FILE=/home/lily/wz336/StructAdapt/outputs/exp-6270/val_outputs/pred_-1.txt
#OUTPUT_FILE=/home/lily/wz336/StructAdapt/evaluation/evaluation/dart-outputs/t5-base.txt
#TEST_TARGETS_REF0=dart_reference/all-reference0.lex
#TEST_TARGETS_REF1=dart_reference/all-reference1.lex
#TEST_TARGETS_REF2=dart_reference/all-reference2.lex
#TEST_TARGETS_DIR=dart_reference
TEST_TARGETS_REF0=/home/lily/wz336/graph2text/test_large_tgt

#bert-score -r ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} -c ${OUTPUT_FILE} --lang en > bertscore.txt
# BLEU

./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF0} < ${OUTPUT_FILE} #> bleu.txt
# OUTPUT_FILE=/home/lily/wz336/graph2text/test_mid1_pred
# TEST_TARGETS_REF0=/home/lily/wz336/graph2text/test_mid1_tgt
# ./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF0} < ${OUTPUT_FILE} #> bleu.txt


OUTPUT_FILE=/home/lily/wz336/graph2text/test_mid_pred
TEST_TARGETS_REF0=/home/lily/wz336/graph2text/test_mid_tgt
./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF0} < ${OUTPUT_FILE} #> bleu.txt

OUTPUT_FILE=/home/lily/wz336/graph2text/test_small_pred
TEST_TARGETS_REF0=/home/lily/wz336/graph2text/test_small_tgt
./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF0} < ${OUTPUT_FILE} #> bleu.txt

#python prepare_files.py ${OUTPUT_FILE} ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2}
# METEOR
#cd meteor-1.5/ 
#java -Xmx2G -jar meteor-1.5.jar ../${OUTPUT_FILE} ../all-notdelex-refs-meteor.txt -l en -norm -r 8 > ../meteor.txt
#cd ..

# TER
#cd tercom-0.7.25/
#java -jar tercom.7.25.jar -h ../relexicalised_predictions-ter.txt -r ../all-notdelex-refs-ter.txt > ../ter.txt
#cd ..

# MoverScore
#python moverscore.py ${TEST_TARGETS_REF0} ${OUTPUT_FILE} > moverscore.txt
# BERTScore
#bert-score -r ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} -c ${OUTPUT_FILE} --lang en > bertscore.txt
# BLEURT
echo xxx
python -m bleurt.score -candidate_file=${OUTPUT_FILE} -reference_file=${TEST_TARGETS_REF0} -bleurt_checkpoint=bleurt/bleurt/test_checkpoint -scores_file=bleurt.txt

#PARENT
python parent_eval/parent_eval.py ${OUTPUT_FILE} ${TEST_TARGETS_DIR} parent_eval/test-dart-reftriples.json > parent.txt

python print_scores.py
