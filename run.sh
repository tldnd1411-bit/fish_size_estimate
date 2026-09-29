python main.py \
        --weights fish_size.pt \
        --tank-id tank-03 \
        --camera-ip 192.168.10.199 \
        --camera-user admin \
        --camera-password "RND_globit9106" \
        --csv results/size_measurement.csv \
        --roi 1500,1000,3000,2160 \
        --calib calibration.npz \
        --overlay-path results/overlay.mp4