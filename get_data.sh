for a_per_poi in 1 2 3 4; do
    for poi_speed in 0.05 0.04 0.03 0.02 0.01 0.0; do
        python3 heterogeneous/play.py \
            --agents_per_poi "$a_per_poi" \
            --poi_speed "$poi_speed" \
            >> output.txt
    done
done