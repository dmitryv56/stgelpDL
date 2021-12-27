#!/usr/bin/env bash

#  stgelpdl.sh -script location in a projects tree
# ../stgelpDL
#    stgelpDL/predictor
#    stgelpDL/msrvcpred
#    stgelpDL/scripts
#             scripts/stgelpdl.sh
#

# Getting stgelpDL-software root path
STGELPDL_SOFTWARE_PATH=$(dirname "${BASH_SOURCE[0]}")"/../"
STGELPDL_SOFTWARE_PATH=$(readlink -f "${STGELPDL_SOFTWARE_PATH}")
export STGELPDL_SOFTWARE_PATH
echo "STGELPDL_SOFTWARE_PATH ${STGELPDL_SOFTWARE_PATH}"

MSRVCPRED_SOFTWARE_PATH=${STGELPDL_SOFTWARE_PATH}/msrvcpred
export MSRVCPRED_SOFTWARE_PATH
echo "MSRVCPRED_SOFTWARE_PATH: ${MSRVCPRED_SOFTWARE_PATH}"

PREDICTOR_SOFTWARE_PATH=${STGELPDL_SOFTWARE_PATH}/predictor
export PREDICTOR_SOFTWARE_PATH
echo "PREDICTOR_SOFTWARE_PATH: ${PREDICTOR_SOFTWARE_PATH}"

MSRVCPRED_SERVER_SERVICE=server_Predictor.py
export MSRVCPRED_SERVER_SERVICE
echo "MSRVCPRED_SERVER_SERVICE: ${MSRVCPRED_SERVER_SERVICE}"

MSRVCPRED_SERVER_SERVICE_PATH=${MSRVCPRED_SOFTWARE_PATH}/app/server/${MSRVCPRED_SERVER_SERVICE}
export MSRVCPRED_SERVER_SERVICE_PATH
echo "MSRVCPRED_SERVER_SERVICE_PATH: ${MSRVCPRED_SERVER_SERVICE_PATH}"


echo ""
echo "PYTHONPATH: ${PYTHONPATH}"

echo $PYTHONPATH | grep -q "${STGELPDL_SOFTWARE_PATH}"
# echo "Result: $?"
if [ $? -ne 1 ]; then
  export PYTHONPATH
else
  if [ -z "$PYTHONPATH" ]; then
    PYTHONPATH=${STGELPDL_SOFTWARE_PATH}
  else
    PYTHONPATH=$PYTHONPATH:${STGELPDL_SOFTWARE_PATH}
  fi
  export PYTHONPYTH
fi

echo ""
echo "PYTHONPATH: ${PYTHONPATH}"

#echo $PYTHONPATH | grep -q "${PREDICTOR_SOFTWARE_PATH}"
## echo "Result: $?"
#if [ $? -ne 0 ]; then
#    PYTHONPATH=${PYTHONPATH}:${PREDICTOR_SOFTWARE_PATH}
#    export PYTHONPYTH
#fi
#
#echo ""
#echo "PYTHONPATH: ${PYTHONPATH}"
#
echo $PYTHONPATH | grep -q "${MSRVCPRED_SOFTWARE_PATH}"
# echo "Result: $?"
if [ $? -ne 0 ]; then
    PYTHONPATH=${PYTHONPATH}:${MSRVCPRED_SOFTWARE_PATH}:${MSRVCPRED_SOFTWARE_PATH}/app:${MSRVCPRED_SOFTWARE_PATH}/src
    export PYTHONPYTH
fi

echo ""
echo "PYTHONPATH: ${PYTHONPATH}"


_get_process_pid() {
    local pattern=${1}
#    echo "(_get_process_pid) pattern: ${pattern}"
#    echo ""
    local out=`ps ax | grep ${pattern} | grep --invert-match "grep ${pattern}"`
#    echo "(_get_process_pid) out: ${out}"
#    echo ""

    if [[ "$out" != "" ]]; then
        out=`echo ${out} | awk '{print $1}'`
        echo -n ${out}
    else
        echo -n 0
    fi
}

is_server_predictor_not_running() {
    local _pid=`_get_process_pid "${MSRVCPRED_SERVER_SERVICE}"`
    echo "status: ${_pid}"
    echo ""
    if [[ ${_pid} -eq 0 ]]; then
        echo "server predictor is not running now"
        echo ""
        return 1
    fi
    # echo "server predictor is already running"
    return 0
}

server_service() {
    local is_back=${1}

    #is_server_predictor_not_running || { echo "ERROR" ; echo "Msrvcpred server Service is Running." ; echo "Stop before start!" ; return ; }

     is_server_predictor_not_running
    ret_code=$?
    echo "ret_code: ${ret_code}"
    if [ ${ret_code} -eq 1 ]; then
      echo ""
      echo "ERROR: Msrvcpred server Service is Running.  Stop before start!"
      echo ""
      return
    fi


    echo ${is_back}
    if [[ "${is_back}" == "run_back" ]]; then
        python3 ${MSRVCPRED_SERVER_SERVICE_PATH} &
        echo $!
    else
        python3 ${MSRVCPRED_SERVER_SERVICE_PATH}
    fi
}

predictor_service() {
  python3 ${STGELPDL_SOFTWARE_PATH}/msrvcpred/src/pred_service.py -m auto -t Imbalance &
  local ret=$!
  echo " Predictor service started... ${ret}"
}