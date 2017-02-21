for i in $(seq 1 3)
do
    echo "*** Stock choice $i ****"
    python linear_hf/choose_100_stocks.py $i
done
