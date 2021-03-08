#!/usr/bin/python3

"""

     chmod +x start_andephyl.py

     ./start_andephyl.py -m debug -M DTWIN -S 64 -s 32 -d "/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log"
     -f 16 -b 256 -R 60 -p 0.3 -c 32 --BFsize 2048 --BFprob 0.05 -z 64 -e 16 -t "CanBus_FD_dbg"

     or

     ./start_andephyl.py --mode debug --method DTWIN --train_size 64 --test_size 32
     --icsim_dump "/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log"
     --fsample 16 --bitrate 256 --SNR 60 --slope 0.3 --chunk 32 --BFsize 2048 --BFprob 0.05 --batch_size 64 epochs 16
     --title "CanBus_FD_dbg"


"""

import sys
from canbus.andephyl import main

if __name__ =="__main__":
    main(len(sys.argv),sys.argv)

    pass