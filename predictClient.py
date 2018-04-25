from grpc.beta import implementations
import numpy
import tensorflow as tf
import config
from PrepareData import reader
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
# 建立连接
channel = implementations.insecure_channel('13.112.30.246', 9000)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'BilstmCRF'
request.model_spec.signature_name = 'serving_default'
dataReader = reader(dict_file=config.dict_file, input_dict = True)
sentence = '我爱北京天安门'
wordIndex = dataReader.sentenceTowordIndex(sentence)
print('wordIndex: {}'.format(wordIndex))
request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(wordIndex, dtype=tf.int32, shape=[1, 64]))
response = stub.Predict.future(request, 5.0)
results = {}
for key in response.result().outputs:
    tensor_proto = response.result().outputs[key]
    nd_array = tf.contrib.util.make_ndarray(tensor_proto)
    results[key] = nd_array
print(results)