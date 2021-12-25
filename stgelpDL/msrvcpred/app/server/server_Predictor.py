#!/usr/bin/env python3

# Copyright 2021 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Python implementation of the GRPC predictor server."""

from concurrent import futures
import logging
from logging.handlers import RotatingFileHandler
import time
from datetime import datetime


import grpc
from io import BytesIO
import microservices_pb2
import microservices_pb2_grpc

# from  msrvcpred.src.readdata import get_data_for_train, get_data_for_predict
from  src.readdata import get_data_for_train, get_data_for_predict
from msrvcpred.cfg import MAX_LOG_SIZE_BYTES, BACKUP_COUNT , PATH_SERVER_LOG, MAGIC_TRAIN_CLIENT, MAGIC_PREDICT_CLIENT,\
    GRPC_PORT,GRPC_IP
from predictor.utility import cSFMT

# set logger

size_handler=RotatingFileHandler(PATH_SERVER_LOG, mode='a', maxBytes =MAX_LOG_SIZE_BYTES, backupCount=BACKUP_COUNT )
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
size_handler.setFormatter(log_formatter)
logger.addHandler(size_handler)

logger=logging.getLogger(__name__)

CRITICAL_MESSAGE ="No data received from data provider!!!!"


class ObtainData(microservices_pb2_grpc.ObtainDataServicer):
    """
        gRPC server for Obtain Data Service
    """
    def __init__(self, *args, **kwargs):
        self.server_port = GRPC_PORT #46001

    def SendData(self, request, context):
        """
                Implementation of the rpc SendData declared in the proto
                file above.
        """
        logger.info("Server: Predict client  {}".format(request))
        list_results = get_data_for_predict(start_time=request.start_time,end_time=request.end_time)
        result = {}
        if list_results is None:
            logger.critical(CRITICAL_MESSAGE)
            timestamp = datetime.now().strftime(cSFMT)
            result = {'timestamp': timestamp, 'real_demand': 0.0, 'programmed_demand': 0.0,
                      'forecast_demand': 0.0, 'status': -1, 'statusmsg': "can not get requested data"}
            logger.error("Server sends {} {}".format(request.clientid, result))
        else:
            result=list_results[-1]  # last item
            logger.info("Server sends {} {}".format(request.clientid,result))

        return microservices_pb2.DataReply(**result)


    def SendStreamData(self, request, context):
        """
        Implementation of the rpc SSendStreamData declared in the proto  file above.
        """
        logger.info("Server: Train client  {}".format(request))


        list_results =get_data_for_train(start_time=request.start_time,end_time=request.end_time)
        if list_results is None:
            logger.critical(CRITICAL_MESSAGE)
            timestamp=datetime.now().strftime(cSFMT)
            result = {'timestamp': timestamp, 'real_demand': 0.0, 'programmed_demand': 0.0,
                      'forecast_demand': 0.0, 'status': -1, 'statusmsg': "can not get requested data"}
            logger.error("Server sends {}: {}".format(request.clientid, result))
            yield microservices_pb2.TrainDataReply(**result)
            return

        for result in list_results:
            logger.info("Server sends {}: {}".format(request.clientid, result))
            yield microservices_pb2.TrainDataReply(**result)

    def start_server(self):
        """
        Function which actually starts the gRPC server, and preps
        it for serving incoming connections
        """
        # declare a server object with desired number
        # of thread pool workers.
        data_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        # This line can be ignored
        microservices_pb2_grpc.add_ObtainDataServicer_to_server(ObtainData(),data_server)

        # bind the server to the port defined above
        data_server.add_insecure_port('[::]:{}'.format(self.server_port))

        # start the server
        data_server.start()
        print ('Data Server running ...')
        logger.info('Data Server running ...')

        try:
            # need an infinite loop since the above
            # code is non blocking, and if I don't do this
            # the program will exit
            while True:
                time.sleep(60*60*60)
        except KeyboardInterrupt:
            data_server.stop(0)
            print('Data Server Stopped ...')
            logger.info('Data Server Stopped ...')

# class ObtainData(microservices_pb2_grpc.ObtainDataServicer):
#     def SendData(self,request,context):
#
#         logger.info("Predict client  {}".format(request.clientid))
#         logger.info(request)
#
#         return microservices_pb2.DataReply(timestamp = "2021-12-17 20:00:01", real_demand = 22.2,
#                                            programmed_demand = 22.3, forecast_demand = 22.4, status = 10,
#                                            statusmsg = "success")
#
#
#
# class ObtainTrainData(microservices_pb2_grpc.ObtainTrainDataServicer):
#     def SendTrainData(self, request, context):
#         logger.info("Train client {}".format(request.clientid))
#         logger.info(request)
#
#         microservices_pb2.TrainDataReply(timestamp = "2021-12-17 21:00:01", real_demand = 22.2,
#                                            programmed_demand = 23.3, forecast_demand = 24.4, status = 20,
#                                            statusmsg = "success")
#         return microservices_pb2.TrainDataReply(timestamp="2021-12-17 21:01:01", real_demand=22.4,
#                                                 programmed_demand=23.4, forecast_demand=24.5, status=21,
#                                                 statusmsg="success")
#
#
# def serve():
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     microservices_pb2_grpc.add_ObtainDataServicer_to_server(ObtainData(),server)
#     microservices_pb2_grpc.add_ObtainTrainDataServicer_to_server(ObtainTrainData(),server)
#     server.add_insecure_port('[::]:50051')
#     server.start()
#     server.wait_for_termination()


if __name__=="__main__":

    # serve()
    curr_server = ObtainData()
    curr_server.start_server()



