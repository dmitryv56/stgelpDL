#!/usr/bin/env python3

""" Due to grpc common issues  (Unavailable : 'failed to connect to all addresses' 'gprc_status':14)
is need to unset http_proxy, https_proxy environment variables.

if os.environ.get('https_proxy'):
 del os.environ['https_proxy']
if os.environ.get('http_proxy'):
 del os.environ['http_proxy']

It is possible add option to client channel

grpc.insecure_channel('localhost:50051', options=(('grpc.enable_http_proxy', 0),))

(see https://stackoverflow.com/questions/57599354/python-not-able-to-connect-to-grpc-channel-failed-to-connect-to-all-addresse )

"""
import sys
import grpc
import logging
# import numpy as np
# from io import BytesIO
import math

from msrvcpred.app import microservices_pb2, microservices_pb2_grpc
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from predictor.utility import cSFMT
from predictor.control import ControlPlane
from msrvcpred.cfg import MAGIC_TRAIN_CLIENT, MAGIC_PREDICT_CLIENT, MAGIC_CHART_CLIENT, MAGIC_WEB_CLIENT, GRPC_PORT, \
    GRPC_IP, ALL_MODELS,DT_DSET, MMV, BREAK_AQUISATION

logger = logging.getLogger(__name__)




class PredictorClient(object):
    """
    Client for accessing the gRPC functionality
    """

    def __init__(self):
        # configure the host and the
        # the port to which the client should connect
        # to.
        self.host = GRPC_IP           # 'localhost'
        self.server_port = GRPC_PORT  # 46001
        self._log = logger
        self.end_time = datetime.now().strftime(cSFMT)
        self.start_time = (datetime.now()-timedelta(days=7)).strftime(cSFMT)

        # instantiate a communication channel
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port), options=(('grpc.enable_http_proxy', 0), ))

        # bind the client to the server channel
        self.stub = microservices_pb2_grpc.ObtainDataStub(self.channel)

    def get_data(self, message):
        """
        Client function to call the rpc for SendData
        """
        # get_data_message =microservices_pb2.DataRequest(clientid=777,region='ElHierro')
        response = self.stub.SendData(microservices_pb2.DataRequest(clientid=MAGIC_PREDICT_CLIENT, region='ElHierro'))
        print(response)
        logger.info(
            "Get predict data: timestamp :{} status:{} real_demand: {}".format(response.timestamp, response.status,
                                                                               response.real_demand))
        return response

    # function to invoke our newly implemented RPC
    def get_streaming_data(self, message) -> (list, list, list, list):
        """
        Client function to call the rpc for SendStreamData
        """
        logger.info(message)

        get_data_message = microservices_pb2.TrainDataRequest(clientid=MAGIC_TRAIN_CLIENT, start_time=self.start_time,
            end_time=self.end_time, region="ElHierro")
        list_requests = self.stub.SendStreamData(get_data_message)
        data_list = []
        dt = []
        real_demand = []
        programmed_demand = []
        forecast_demand = []

        for request in list_requests:
            print(request)

            ret = self.request_parse(request, dt, real_demand, programmed_demand, forecast_demand)

            if ret == BREAK_AQUISATION:
                break

        return dt, real_demand, programmed_demand, forecast_demand

    """ Parse request (TrainDataReply or DataReply, see in microservices_pb2_gprc.py).
    Parsing process is broken when 'real_demand' missed in the request.
    """
    def request_parse(self, request: object, dt: list, real_demand: list, programmed_demand: list,
                      forecast_demand: list) -> int:

        list_attr = dir(request)
        if 'status' in list_attr:
            status = request.status
            if status < 0:
                self._log.error("Error status {}: Ignore request: {}".format(status, request))
                return status

        if 'real_demand' in list_attr:
            real_demand.append(request.real_demand)
        else:
            self._log.error("For {} real_demand missed. Break data aquisation  on {}".format(status, request))
            return BREAK_AQUISATION
        if 'timestamp' in list_attr:
            dt.append(request.timestamp)
        if 'programmed_demand' in list_attr:
            programmed_demand.append(request.programmed_demand)
        if 'forecast_demand' in list_attr:
            forecast_demand.append(request.forecast_demand)
        smsg = "{:<5d} {:<32s} {:>15.5f} {:>15.5f} {:>15.5f}".format(status, request.timestamp, request.real_demand,
                                                                     request.programmed_demand,
                                                                     request.forecast_demand)
        self._log.info(smsg)
        return status

    def save_titles(self, message):
        """
        Client function for SaveTitle -rpc
        """
        (ts_imbalace_name, ts_programmed_name, ts_demand_name) = ControlPlane.get_modeImbalanceNames()

        l_models = ["" for i in range(10)]  # TODO MAX_NUM_MODELS
        for key,val in ALL_MODELS.items():
            for item in val:
                (num,name)=item
                l_models[num]=name

        model0 = l_models[0]
        model1 = l_models[1]
        model2 = l_models[2]
        model3 = l_models[3]
        model4 = l_models[4]
        model5 = l_models[5]
        model6 = l_models[6]
        model7 = l_models[7]
        model8 = l_models[8]
        model9 = l_models[9]

        self._log.info(
            "Store header names: client: {} timestamp :{} timeseries: {} models: {} {} {} {} {} {} {} {} {} {}".format(
                MAGIC_PREDICT_CLIENT, DT_DSET, ts_imbalace_name, model0, model1, model2, model3, model4, model5, model6,
                model7, model8, model9))
        response = self.stub.SaveTitle(microservices_pb2.SaveTitleRequest(clientid=MAGIC_PREDICT_CLIENT,
                    timestamp=DT_DSET, timeseries=ts_imbalace_name, model0=model0, model1=model1, model2=model2,
                    model3=model3, model4=model4, model5=model5, model6=model6, model7=model7, model8= model8,
                    model9=model9))
        self._log.info(response)
        logger.info(
            "Store header names status {}".format(response.status))


        return response

    def get_titles(self, message) -> dict:
        self._log.info("{}-client: {}".format(MAGIC_CHART_CLIENT,message))
        d_res = {}
        try:
            response = self.stub.GetTitle(microservices_pb2.GetTitleRequest(clientid=MAGIC_CHART_CLIENT))
        except Exception as ex:
            self._log.critical("O-oops! Exception raised! {}".format(ex))
            self._log.info("Check server is running! Chart client finished.")
            sys.exit(0)

        if response.status !=0 :
            self._log.error("can not get model names from server")
        else:
            d_res['x_label']=response.timestamp
            d_res['models'] = [response.timeseries, response.model0,response.model1, response.model2, response.model3,
                                response.model4, response.model5, response.model6, response.model7, response.model8,
                                response.model9 ]
            self._log.info("Labels x,y before empty string filtering: {}".format(d_res))
            d_res['models'][:] = [x for x in d_res['models'] if x]
            self._log.info("Labels x,y : {}".format(d_res))
        return d_res

    def save_predicts(self, message):
        """
        Client function for SavePredictsRequest -rpc
        The current predict data has not timeseries value because it does not exist at predict time. Instead, the
        previous timeserires value is sent. The server writes it in previous predict and current predict timeseries
        value sets to MMV (magic missed value)

        """

        df = pd.read_csv(message)
        (n, m) = df.values.shape
        self._log.info("DataFrame shape: ({},{})".format(n,m))
        last_values = df.values[-1]
        self._log.info("last_values:  {}".format(last_values))

        last_values[1] = MMV if math.isnan(last_values[1]) else last_values[1]

        if n >= 2:
            pred_last_values = df.values[-2]
            self._log.info("pred last values:  {}".format(pred_last_values))
            last_values[1] =MMV  if math.isnan(pred_last_values[1]) else pred_last_values[1]

        clientid = MAGIC_PREDICT_CLIENT
        timestamp = last_values[0]
        timeseries = last_values[1]
        model0 = last_values[0 + 2]
        model1 = last_values[1 + 2]
        model2 = last_values[2 + 2]
        model3 = last_values[3 + 2]
        model4 = last_values[4 + 2]
        model5 = last_values[5 + 2]
        model6 = last_values[6 + 2]
        model7 = last_values[7 + 2]
        model8 = 1e-15
        model9 = 1e-15
        self._log.info(clientid, timestamp, timeseries, model0, model1, model2, model3, model4, model5, model6,
                       model7, model8, model9)
        response = self.stub.SavePredicts(microservices_pb2.SavePredictsRequest(clientid=clientid,
                timestamp=timestamp, timeseries=timeseries, model0=model0, model1=model1, model2=model2, model3=model3,
                model4=model4, model5=model5, model6=model6, model7=model7, model8=model8, model9=model9))

        self._log.info(response)
        logger.info(
            "Store predicts status {}".format(response.status))

        return response

    # function to invoke our newly implemented RPC
    def get_predicts(self, message) -> (list, dict):
        """
            Client function to call the rpc for SendStreamData
        """
        logger.info("{} -client : {}".format(MAGIC_CHART_CLIENT,message))

        get_predicts_message = microservices_pb2.PredictsDataRequest (clientid=MAGIC_CHART_CLIENT)
        list_requests = self.stub.GetPredicts(get_predicts_message)

        timestamp = []
        timeseries = []
        model0 = []
        model1 = []
        model2 = []
        model3 = []
        model4 = []
        model5 = []
        model6 = []
        model7 = []
        model8 = []
        model9 = []

        d_actual_data = {}

        for request in list_requests:
            self._log.info(request)

            if request.status != 0:
                self._log.error("The server did not send predict data: {}".format(request))
                break

            timestamp.append(request.timestamp)

            aux_list = [ timeseries, model0, model1, model2, model3, model4, model5, model6, model7, model8, model9]
            req_aux_list = [request.timeseries, request.model0, request.model1, request.model2, request.model3,
                            request.model4, request.model5, request.model6, request.model7, request.model8,
                            request.model9]

            i = 0
            for (item, req_item)  in zip(aux_list, req_aux_list):
                item.append(req_item if abs(req_item)>2.0*MMV else float('NaN'))
                if i == 0:  # timeseries may content NaN
                    d_actual_data[str(i)] = item
                    i = i + 1
                else:
                    cleanlist = [x for x in item if x == x]
                    if cleanlist:
                        d_actual_data[str(i)] = item
                        i = i + 1


            # timeseries.append(request.timeseries if abs(request.timeseries)>2.0*MMV else float('NaN'))
            # cleanlist = [x for x in timeseries if x == x]
            # if cleanlist:  d_actual_data[str(i)] = timeseries; i=i+1
            # model0.append(request.model0 if abs(request.model0)>2.0*MMV else float('NaN'))
            # cleanlist = [x for x in model0 if x == x]
            # if cleanlist:  d_actual_data[str(i)] = model0; i=i+1
            # model1.append(request.model1 if abs(request.model1)>2.0*MMV else float('NaN'))
            # cleanlist = [x for x in model1 if x == x]
            # if cleanlist:  d_actual_data[str(i)] = model1; i=i+1
            # model2.append(request.model2 if abs(request.model2)>2.0*MMV else float('NaN'))
            # cleanlist = [x for x in model2 if x == x]
            # if cleanlist:  d_actual_data[str(i)] = model2; i=i+1
            # model3.append(request.model3 if abs(request.model3)>2.0*MMV else float('NaN'))
            # cleanlist = [x for x in model3 if x == x]
            # if cleanlist:  d_actual_data[str(i)] = model3; i=i+1
            # model4.append(request.model4 if abs(request.model4)>2.0*MMV else float('NaN'))
            # cleanlist = [x for x in model4 if x == x]
            # if cleanlist:  d_actual_data[str(i)] = model4; i=i+1
            # model5.append(request.model5 if abs(request.model5)>2.0*MMV else float('NaN'))
            # cleanlist = [x for x in model5 if x == x]
            # if cleanlist:  d_actual_data[str(i)] = model5; i=i+1
            # model6.append(request.model6 if abs(request.model6)>2.0*MMV else float('NaN'))
            # cleanlist = [x for x in model6 if x == x]
            # if cleanlist:  d_actual_data[str(i)] = model6; i=i+1
            # model7.append(request.model7 if abs(request.model7)>2.0*MMV else float('NaN'))
            # cleanlist = [x for x in model7 if x == x]
            # if cleanlist:  d_actual_data[str(i)] = model7; i=i+1
            # model8.append(request.model8 if abs(request.model8)>2.0*MMV else float('NaN'))
            # cleanlist = [x for x in model8 if x == x]
            # if cleanlist:  d_actual_data[str(i)] = model8; i=i+1
            # model9.append(request.model9 if abs(request.model9)>2.0*MMV else float('NaN'))
            # cleanlist = [x for x in model9 if x == x]
            # if cleanlist:  d_actual_data[str(i)] = model9; i=i+1

        self._log.info("Timestamp: {}".format(timestamp))
        self._log.info("Timeseries: {}".format(timeseries))
        self._log.info("Model0: {}".format(model0))
        self._log.info("Model1: {}".format(model1))
        self._log.info("Model2: {}".format(model2))
        self._log.info("Model3: {}".format(model3))
        self._log.info("Model4: {}".format(model4))

        # i=0
        # cleanlist = [x for x in timeseries if x == x]
        # if cleanlist:  d_actual_data[str(i)] = timeseries; i=i+1
        # cleanlist = [x for x in model0 if x == x]
        # if cleanlist:  d_actual_data[str(i)] = model0; i = i + 1
        # cleanlist = [x for x in model1 if x == x]
        # if cleanlist:  d_actual_data[str(i)] = model1; i = i + 1
        # cleanlist = [x for x in model2 if x == x]
        # if cleanlist:  d_actual_data[str(i)] = model2; i = i + 1
        # cleanlist = [x for x in model3 if x == x]
        # if cleanlist:  d_actual_data[str(i)] = model3; i = i + 1
        # cleanlist = [x for x in model4 if x == x]
        # if cleanlist:  d_actual_data[str(i)] = model4; i = i + 1
        # cleanlist = [x for x in model5 if x == x]
        # if cleanlist:  d_actual_data[str(i)] = model5; i = i + 1
        # cleanlist = [x for x in model6 if x == x]
        # if cleanlist:  d_actual_data[str(i)] = model6; i = i + 1
        # cleanlist = [x for x in model7 if x == x]
        # if cleanlist:  d_actual_data[str(i)] = model7; i = i + 1
        # cleanlist = [x for x in model8 if x == x]
        # if cleanlist:  d_actual_data[str(i)] = model8; i = i + 1
        # cleanlist = [x for x in model9 if x == x]
        # if cleanlist:  d_actual_data[str(i)] = model9; i = i + 1

        self._log.info("Actual predict data: {}".format(d_actual_data))

        return timestamp, d_actual_data


        # def run():
#
#     with grpc.insecure_channel('127.0.0.1:50051') as channel:
#         stub = microservices_pb2_grpc.ObtainDataStub(channel)
#         response = stub.SendData(microservices_pb2.DataRequest(clientid=777, region="ElHierro"))
#         print("client received: {}".format(response))
#         logger.info("client 777   received: {}".format(response))
#


if __name__ == '__main__':
    currs_client = PredictorClient()
    logger.info(currs_client.get_data("Predict data"))

    while True:
        # run()
        # time.sleep(1)
        logger.info("Train data\n\n\n\n")
        time.sleep(2)
        currs_client.get_streaming_data("Train data")
        time.sleep(5)
        logger.info("Predict data\n\n\n\n")
        for i in range(5):
            logger.info(currs_client.get_data("Predict data"))
            time.sleep(2)

