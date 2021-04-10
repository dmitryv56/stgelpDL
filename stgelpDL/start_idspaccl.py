#!/usr/bin/python3

""" 'idspaccl' - there is a digital twin of CANbus-FD protocol, it performs for intrusion detection in the packets
at the transport level of Can Bus-FD. The Neural Net is used for packet classification of two groups  valid and
intrused packets.
    At train stage, the packets are classified by given matched key and the probability of the class appearance is estimated.
The given matched key is one of packet fields 'ID', 'Data' or concatenation 'ID'+'Data' ('Packet').
The class probability estimation and a histogram of the delay intervals between packets being belonging to to the same
class are saved in Database (pandas DataFrame dataset).
    At the test stage, anomaly packed are detected. The checked hypothesis is packet appearances describes by poisson
probability distributions  and delay intervals describes by exponential distribution. For all classes the parameters of
distributions already estimated .

    The stand-alone script for the digital twin launching should be 'executable'

     chmod +x start_idspaccl.py

    To run it the foliowing parameters are passed via command line:

     ./start_idspaccl.py -m debug -M ids -k Packet -S 60000 -s 20000
     -d "/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log" -c 32 -t "IDS_classification"

     or

     ./start_idspaccl.py --mode debug --method ids  --train_size 60000 --test_size 20000
     --icsim_dump "/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log"
     --chunk 32 --BFsize 2048 --BFprob 0.05   --title "IDS_classification"

     where '-m' or '--mode' one of {'train','test' ,'debug'}. In 'debug' mode packet sequence is split on tran and test
            sequences.
            '-M' or '--method' - now only 'ppd' -the poisson probability distribution.
            '-d' or '--icsim_dump' dump file of ICsim , the source of packet sequence.
            '-S' or '--train_size' size of training sequence, only for 'debug' mode.
            '-s' or '--test_size' size ot evaluation or test sequence, only for 'debug' mode/
            '-c' or '--chunk' size of chunk for reading dump file.
            '-t' or '--title' one world for log,repository name.

    To read help information
    ./start_idspaccl.py --help
    ./start_idspaccl.py -h

    To fet version
    ./start_idspaccl.py --version.

"""

import sys
from canbus.idspaccl import main

if __name__ =="__main__":
    nret = main(len(sys.argv),sys.argv)

    sys.exit(nret)