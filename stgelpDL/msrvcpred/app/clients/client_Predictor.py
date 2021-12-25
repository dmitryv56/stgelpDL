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

import grpc
import logging
import numpy as np
from io import BytesIO
import microservices_pb2
import microservices_pb2_grpc
import time
from datetime import datetime, timedelta

from predictor.utility import cSFMT
from msrvcpred.cfg import MAGIC_TRAIN_CLIENT, MAGIC_PREDICT_CLIENT,GRPC_PORT,GRPC_IP



logger =logging.getLogger(__name__)

BREAK_AQUISATION = -999

class PredictorClient(object):
    """
    Client for accessing the gRPC functionality
    """

    def __init__(self):
        # configure the host and the
        # the port to which the client should connect
        # to.
        self.host = GRPC_IP          # 'localhost'
        self.server_port = GRPC_PORT # 46001
        self._log = logger
        self.end_time = datetime.now().strftime(cSFMT)
        self.start_time=(datetime.now()-timedelta(days=7)).strftime(cSFMT)

        # instantiate a communication channel
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port), options=(('grpc.enable_http_proxy',0),))

        # bind the client to the server channel
        self.stub = microservices_pb2_grpc.ObtainDataStub(self.channel)

    def get_data(self, message):
        """
        Client function to call the rpc for SendData
        """
        # get_data_message =microservices_pb2.DataRequest(clientid=777,region='ElHierro')
        response = self.stub.SendData(microservices_pb2.DataRequest(clientid=MAGIC_PREDICT_CLIENT,region='ElHierro'))
        print(response)
        logger.info("Get predict data: timestamp :{} status:{} real_demand: {}".format(response.timestamp,
                                                                        response.status,response.real_demand))
        return response

    # function to invoke our newly implemented RPC
    def get_streaming_data(self, message)->(list,list,list,list):
        """
        Client function to call the rpc for SendStreamData
        """
        logger.info(message)

        get_data_message =microservices_pb2.TrainDataRequest(clientid=MAGIC_TRAIN_CLIENT, start_time=self.start_time,
                                                            end_time=self.end_time,region="ElHierro")
        list_requests = self.stub.SendStreamData(get_data_message)
        data_list=[]
        dt=[]
        real_demand=[]
        programmed_demand=[]
        forecast_demand=[]

        for request in list_requests:
            print(request)

            ret = self.request_parse(request,dt, real_demand, programmed_demand,forecast_demand)

            if ret == BREAK_AQUISATION:
                break


        return dt, real_demand,programmed_demand, forecast_demand
    """ Parse request (TrainDataReply or DataReply, see in microservices_pb2_gprc.py).
    Parsing process is broken when 'real_demand' missed in the request.
    """
    def request_parse(self,request:object,dt:list, real_demand:list, programmed_demand:list,forecast_demand:list)->int:

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

