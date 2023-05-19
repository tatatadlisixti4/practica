import os
import os.path as osp
import pathlib
import cv2
import argparse
import numpy as np
import time
from tensorflow import keras
from keras.utils import to_categorical
import tensorrt as trt
from TensorRT import common

IMG_SIZE = 128
#IMG_SIZE = 64
TRT_LOGGER = trt.Logger() # Utilizado para registrar mensajes e información durante la ejecución de TensorRT

def lectura_datos(data_dir):
    #Datos del modelo
    dataset = '850'
    dataset_variant = 'normal'
    hands = ['Left', 'Right']
    nclases = 100
    mode = 'all'
    nsamples = 6
    
    #Tipos de bases de datos
    db = {
        "850": {"percent": 0.8, "ext": "jpg"},
        "CASIA/940": {"percent": 0.8, "ext": "jpg"},
        "FYO": {"percent": 0.0, "ext": "png"},
        "IIT": {"percent": 0.5, "ext": "png"},
        "POLYU": {"percent": 0.6, "ext": "jpg"},
        "PUT": {"percent": 0.8, "ext": "jpg"},
        "TONGJI": {"percent": 0.8, "ext": "png"},
        "VERA": {"percent": 0.8, "ext": "png"},
        "NS-PVDB": {"percent": 0.8, "ext": "png"},
        "Synthetic-sPVDB": {"percent": 0.8, "ext": "png"}
    }
    if dataset in db: 
            percent, ext = db[dataset]["percent"], db[dataset]["ext"]
    
    #Creación de las clases
    clases = []
    if len(hands) == 0:
        for i in range(1, nclases+1):
            c = str(i).zfill(5)
            clases.append(c)
    else:
        for i in range(1, nclases+1):
            for h in hands:
                c = str(i).zfill(3)
                clases.append(osp.join(c, h))          
    nclases = len(clases)
    
    #Cración lista con las imagenes a utilizar
    test_data = []
    if mode == "ft+aug":
        pass
    elif mode == "ft":
        pass
    else:
        for class_num in range(nclases): 
            for s in range(int(nsamples*percent)+1, nsamples+1):
                if (class_num) %2 == 1:
                    imagen = str(class_num//2 +1).zfill(3) + '_r_' + dataset + '_0' + str(s) + '_rot0_cx0_cy0' + '.' + ext
                    #print(class_num, imagen)
                else:
                    imagen = str(class_num//2 +1).zfill(3) + '_l_' + dataset + '_0' + str(s) + '_rot0_cx0_cy0' + '.' + ext
                    #print(class_num, imagen)
                    
                aux = osp.join(data_dir, dataset, imagen)
                img_array = cv2.imread(aux, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([new_array, class_num])

    return test_data, nclases

def procesamiento_datos(test_data, nclases):
    mode = 'all'
    test_samples = []
    test_labels = []

    for s, l in test_data:
        test_samples.append(s)
        test_labels.append(l)
    
    #Procesamiento 1
    test_samples = np.array(test_samples).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_labels = np.array(test_labels)
    
    #Procesamiento 2
    meanTest = np.mean(test_samples, axis=0)
    test_samples = test_samples-meanTest
    test_samples = test_samples/255

    #Procesamiento 3
    test_samples = test_samples.reshape(test_samples.shape[0], IMG_SIZE, IMG_SIZE, 1)
    test_labels = to_categorical(test_labels, nclases)
    
    return test_samples, test_labels 

def get_engine(onnx_file_path, engine_file_path=""):
    if os.path.exists(engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
        
def test(model_path, test_samples, test_labels):
    with get_engine("no", model_path) as engine, engine.create_execution_context() as context:
        for imagen in test_samples:
            inputs_trt, outputs_trt, bindings_trt, stream_trt = common.allocate_buffers(engine,imagen)
            start = time.time()
            inputs_trt[0].host = imagen
            results = common.do_inference_v2(context, bindings=bindings_trt, inputs=inputs_trt, outputs=outputs_trt, stream=stream_trt)
            ms = time.time() - start
            results = np.squeeze(results)
            predicted_label = np.argmax(results)
            score = results[predicted_label]


if __name__ == "__main__":
    #Llamada función lectura de datos con el directorio de la bd
    data_dir = osp.join(pathlib.Path().parent.absolute() , 'datasets', 'CASIA')
    test_data, nclases = lectura_datos(data_dir)

    #Procesamiento datos
    test_samples, test_labels = procesamiento_datos(test_data, nclases)

    #Creación estructura 
    parser = argparse.ArgumentParser(description='Test a CNN')
    parser.add_argument('--model_path', type=str, required=False, default="/home/rhernandez/modelos_trt/casia_batch16_fp16.trt", help="Path to the trained model")
    args = parser.parse_args()
    model_path = args.model_path

    if os.path.exists(model_path):
        print('Existe')
        test(model_path, test_samples, test_labels)
    else:
        print('No existe la ruta', model_path)