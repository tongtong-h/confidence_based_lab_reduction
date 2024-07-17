for kf in {0..4}
do
    sed -i "s/kfold: .*/kfold: ${kf},/" config_ts_baseline_panel_loss.cfg
    for c in $(seq 0.75 0.05 1.0)
    do
        echo "kfold = ${kf}"
        echo "coverage = ${c}"
        sed -i "s/coverage: .*/coverage: ${c},/" config.cfg
        python main.py --cfg config.cfg
    done
done
