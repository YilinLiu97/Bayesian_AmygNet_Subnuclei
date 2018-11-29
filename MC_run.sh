for i in {1..10}; do python net_segment.py inference -c config/AmygNet_config.ini --keep_prob=0.7 --output_prob=True --save_seg=MC/subj_ID/run_$i;done

python MC.py [file_that_provides_affine] MC/subj_ID # for generating the uncertainty map and MC segmentation result
