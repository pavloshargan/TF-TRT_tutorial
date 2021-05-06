import argparse
import numpy as np
import tensorrt as trt
import time

import pycuda.driver as cuda
import pycuda.autoinit

import cv2

LOGGER = trt.Logger(trt.Logger.WARNING)
DTYPE = trt.float32

# Model
TRT_ENGINE = './engine.trt'
INPUT_NAME = 'input_1:0'
INPUT_SHAPE = (3, 224, 224)

# how many video frames we are going to analyze
LOOP_TIMES = 500

VIDEO_PATH = "./record.avi"

def allocate_buffers(engine):
    #uncomment the code if you have 2 outputs
    print('allocate buffers')
    print(engine.get_binding_shape(0))
    print(engine.get_binding_shape(1))
    # print(engine.get_binding_shape(2))
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
    h_output1 = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
    # h_output2 = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(2)), dtype=trt.nptype(DTYPE))

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output1 = cuda.mem_alloc(h_output1.nbytes)
    # d_output2 = cuda.mem_alloc(h_output2.nbytes)


    #return h_input, d_input, h_output1, h_output2, d_output1, d_output2
    return h_input, d_input, h_output1, d_output1


def load_input(img_, host_buffer):
    print('load input')
    
    c, h, w = INPUT_SHAPE
    dtype = trt.nptype(DTYPE)
    
    img = cv2.resize(img_, (h,w), cv2.INTER_LINEAR)
    #add image preprocessing here if needed

    img_array = img.ravel()

    np.copyto(host_buffer, img_array)


def do_inference(n, context, h_input, d_input, h_output1, h_output2, d_output1, d_output2):
    # Transfer input data to the GPU.
    cuda.memcpy_htod(d_input, h_input)

    # Run inference.
    st = time.time()
    #context.execute(batch_size=1, bindings=[int(d_input), int(d_output1), int(d_output2)])
    context.execute(batch_size=1, bindings=[int(d_input), int(d_output1)])
    print('Inference time {}: {} [msec]'.format(n, (time.time() - st)*1000))

    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh(h_output1, d_output1)
    #cuda.memcpy_dtoh(h_output2, d_output2)

    #return h_output1, h_output2
    return h_output1


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(TRT_ENGINE, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        #h_input, d_input, h_output1, h_output2, d_output1, d_output2 = allocate_buffers(engine)
        h_input, d_input, h_output1, d_output1 = allocate_buffers(engine)
   
    with engine.create_execution_context() as context:
        loop_start = time.time()
        for i in range(LOOP_TIMES):
            ret, img = cap.read()
            load_input(img, h_input)
            #output = do_inference(i, context, h_input, d_input, h_output1, h_output2, d_output1, d_output2)
            output = do_inference(i, context, h_input, d_input, h_output1, d_output1)
            
            print("output: ", output)

            # cv2.imshow("frame",img)
            # k = cv2.waitKey(1) & 0xff
            # if k == 27:
            #     cv2.destroyAllWindows()
            #     sys.exit()

        print("avg_time: ", (time.time()-loop_start)/LOOP_TIMES)


                
if __name__ == '__main__':
    main()