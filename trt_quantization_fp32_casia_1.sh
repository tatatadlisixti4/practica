/usr/src/tensorrt/bin/trtexec --onnx=/home/rhernandez/onnx_models/casia.onnx --saveEngine=/home/rhernandez/modelos_trt/casia_batch1_fp32.trt
/usr/src/tensorrt/bin/trtexec --loadEngine=/home/rhernandez/modelos_trt/casia_batch1_fp32.trt > casia_batch1_fp32.txt