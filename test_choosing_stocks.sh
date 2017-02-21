for i in $(seq 1 3)
do
    echo "*** Stock choice $i ****"
    python -m linear_hf.choose_100_stocks $i
done
