# Building a real-time button detector using Nvidia DeepStream
This file gives the instruction to train a button detector using yoloV3 and deploy the program in deepstream with live video from realsense D435.

## Part 1: Training


## Part 2: Deployment

We take the trained model weight from part 1 and deploy it on an edge device. We explain how to deploy on a Jetson AGX Xavier device using the DeepStream SDK, but you can deploy on any NVIDIA-powered device, from embedded Jetson devices to large datacenter GPUs such as T4.

### Prerequisites
This post uses the following resources:

- A Jetson AGX Xavier device. The instructions can be used on any Jetson devices or any datacenter GPU. 
- The DeepStream SDK for real-time, video analytic application with TensorRT for deep learning inference. Download the source code [here](https://developer.nvidia.com/deepstream-getting-started) and unzip the file. In our case, we have a root folder named `deepstream_sdk_v4.0.2_jetson`. If you want to test yolo with weight trained on COCO, please refer to the README in current folder. Otherwise, follow the instructions below to deploy your customized weight.

Place the weights and cfg files from Part 1 under `deepstream_sdk_v4.0.2_jetson/sources/objectDetector_Yolo/`

### Build the yoloV3 custom parser
DeepStream uses TensorRT for inference. With DeepStream, you have an option of either providing the weights and cfg file directly or providing the TensorRT .engine file. A TensorRT plan file was generated in the Build TensorRT engine step, so use that with DeepStream.

TensorRT takes the input tensors and generates output tensors. To detect an object, tensors from the output layer must be converted to X,Y coordinates or the location of the bounding box of the object. This step is called bounding box parsing. 

```bash
cd deepstream_sdk_v4.0.2_jetson/sources/objectDetector_Yolo/objectDetector_Yolo
```

By default, DeepStream ships with built-in parsers for yoloV3 in `deepstream_sdk_v4.0.2_jetson/sources/objectDetector_Yolo/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp`. Modify line 32 to
```
// change the number of classes here
static const int NUM_CLASSES_YOLO = 3
```

```bash
cd ..
export CUDA_VER=10.0
make -C nvdsinfer_custom_impl_Yolo
```

### Create configuration
 1. Create deepstream_app_config_yoloV3-D435.txt
    - Realsense D435 streams 3 videos under /dev/video0/1/2. video2 is the RGB coloe iamge stream. You can use `v4l2-ctl --list-formats-ext --device video2` to list information of the stream. Set `camera-v4l2-dev-node=1` under [source0]
    - If deployed on TX2, you have to inference every few frame in order to have realtime detection. By default, DeepStream runs inference every frame. If you run inference at 30 fps, the GPU has to do 30 inference operations per second. Depending on the size of the model and the size of the GPU, this might exceed the computing capacity. For example, the process rate of TX2 for yoloV3 is 5 fps, you need to set `interval=6` under [primary-gie] to have good realtime detection.
    - Use a high-quality tracker to predict the bounding box of the object based on previous locations. The tracker is generally less compute-intensive than doing a full inference. With a tracker, you can process more streams or even do a higher resolution inference.
    ```
    [application]
    enable-perf-measurement=1
    perf-measurement-interval-sec=5
    #gie-kitti-output-dir=streamscl

    [tiled-display]
    enable=1
    rows=1
    columns=1
    width=1280
    height=720

    [source0]
    enable=1
    #Type - 1=CameraV4L2 2=URI 3=MultiURI
    type=1
    camera-width=1280
    camera-height=720
    camera-fps-n=30
    camera-fps-d=1
    camera-v4l2-dev-node=2

    [sink0]
    enable=1
    #Type - 1=FakeSink 2=EglSink 3=File
    type=2
    sync=1
    display-id=0
    offset-x=0
    offset-y=0
    width=0
    height=0
    overlay-id=1
    source-id=0

    [osd]
    enable=1
    gpu-id=0
    border-width=1
    text-size=15
    text-color=1;1;1;1;
    text-bg-color=0.3;0.3;0.3;1
    font=Serif
    show-clock=0
    clock-x-offset=800
    clock-y-offset=820
    clock-text-size=12
    clock-color=1;0;0;0
    nvbuf-memory-type=0

    [streammux]
    gpu-id=0
    ##Boolean property to inform muxer that sources are live
    live-source=1
    batch-size=1
    ##time out in usec, to wait after the first buffer is available
    ##to push the batch even if the complete batch is not formed
    batched-push-timeout=40000
    ## Set muxer output width and height
    width=1280
    height=720
    ##Enable to maintain aspect ratio wrt source, and allow black borders, works
    ##along with width, height properties
    enable-padding=0
    nvbuf-memory-type=0

    # config-file property is mandatory for any gie section.
    # Other properties are optional and if set will override the properties set in
    # the infer config file.
    [primary-gie]
    enable=1
    gpu-id=0
    model-engine-file=model_b1_int8.engine
    labelfile-path=labels-control.txt
    batch-size=1
    #Required by the app for OSD, not a plugin property
    bbox-border-color0=1;0;0;1
    bbox-border-color1=0;1;1;1
    bbox-border-color2=0;0;1;1
    bbox-border-color3=0;1;0;1
    #interval=0
    gie-unique-id=1
    nvbuf-memory-type=0
    config-file=config_infer_primary_yoloV3-control.txt

    [tests]
    file-loop=0
    ```

2. Create config_infer_primary_yoloV3-control.txt
    ```
    [property]
    gpu-id=0
    net-scale-factor=1
    #0=RGB, 1=BGR
    model-color-format=0
    custom-network-config=yolov3-control_chess.cfg
    model-file=yolov3-control_chess_best.weights
    #model-engine-file=model_b1_int8.engine
    labelfile-path=labels-control.txt
    int8-calib-file=yolov3-calibration.table.trt5.1
    ## 0=FP32, 1=INT8, 2=FP16 mode
    network-mode=1
    num-detected-classes=3
    gie-unique-id=1
    is-classifier=0
    maintain-aspect-ratio=1
    parse-bbox-func-name=NvDsInferParseCustomYoloV3
    custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
    ```
3. Create labels-control.txt
    ```
    Stop
    Pause
    Start
    ```

### Run the application
```bash
deepstream-app -c deepstream_app_config_yoloV3-camera.txt
```

You should see a video window. And in the terminal you should see the performance 
```
**PERF: 16.19 (16.19)	
**PERF: 16.33 (16.26)	
**PERF: 16.23 (16.25)	
**PERF: 16.39 (16.29)	
**PERF: 18.72 (16.79)	
**PERF: 18.94 (17.15)	
**PERF: 18.99 (17.42)	
**PERF: 18.89 (17.61)	
**PERF: 18.36 (17.70)	
**PERF: 17.93 (17.72)	
**PERF: 18.69 (17.81)	
**PERF: 18.70 (17.88)	
**PERF: 18.73 (17.95)	
**PERF: 18.70 (18.00)	
**PERF: 18.68 (18.05)	
**PERF: 18.75 (18.09)	
**PERF: 17.94 (18.08)	
**PERF: 18.40 (18.10)	
**PERF: 16.22 (18.00)	

**PERF: FPS 0 (Avg)	
**PERF: 14.98 (17.85)	
**PERF: 16.22 (17.77)	
**PERF: 17.50 (17.76)	
**PERF: 18.19 (17.78)	
**PERF: 18.50 (17.81)	
**PERF: 18.63 (17.84)	
**PERF: 17.72 (17.84)	
**PERF: 18.57 (17.86)	
**PERF: 18.65 (17.89)	
**PERF: 18.35 (17.91)	
```