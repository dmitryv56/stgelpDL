#!/usr/bin/python3

from os import path
import sys
from datetime import datetime
from pathlib import Path


from canbus.drive import train_path,test_path
from canbus.BF import BF
from clustgelDL.auxcfg  import D_LOGS,listLogSet, closeLogs,log2All,exec_time,logList
from predictor.utility import msg2log,PlotPrintManager

def main(argc,argv)->int:
    now = datetime.now()
    date_time = now.strftime("%d_%m_%y__%H_%M_%S")
    message1 = "Time execution logging started at {}\n\n".format(datetime.now().strftime("%d %m %y %H:%M:%S"))

    title="canbusFD_train"
    dir_path = path.dirname(path.realpath(__file__))
    folder_for_logging = Path(dir_path) / "Logs" / "{}_{}".format(title, date_time)
    folder_repository=Path(dir_path) / "Repository"
    Path(folder_repository).mkdir(parents=True, exist_ok=True)
    listLogSet(str(folder_for_logging))  # A logs are creating

    method="BF" # BF -Bloom Filter, DTWIN - digital twin
    method = "DTWIN"
    canbusdump ="/home/dmitryv/ICSim/ICSim/candump-2021-02-23_143436.log"
    fsample= 16e+6
    bitrate: float = 125000.0
    slope: float = 0.3
    snr:float = 40.0
    chunk_size=32
    bf = BF(filter_size=1024, fp_prob=0.05, f=D_LOGS['block'])

    train_path(method=method, canbusdump=canbusdump, bf=bf, chunk_size=chunk_size, fsample=fsample,
                bitrate=bitrate, slope=slope, snr=snr, repository=str(folder_repository),f=D_LOGS['control'] )
    # canbusdump ="/home/dmitryv/ICSim/ICSim/candump - 2021 - 02 - 11_100040.log"
    test_path(method=method,canbusdump=canbusdump, bf=bf, chunk_size=chunk_size, fsample=fsample,
                bitrate=bitrate, slope=slope, snr=snr, repository=str(folder_repository),f=D_LOGS['control'])

    closeLogs()
    return 0


if __name__=="__main__":
    nret =main(len(sys.argv),sys.argv)

    sys.exit(nret)