for i in $(seq 1 50)
do
    echo "*** This is run $i ****"
    python -m linear_hf.hyperparameters
done
