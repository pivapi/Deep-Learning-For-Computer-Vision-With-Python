1 # import the necessary packages
2 from config import car_config as config
3 import mxnet as mx
4 import argparse
5 import logging
6 import os
7 8
# construct the argument parse and parse the arguments
9 ap = argparse.ArgumentParser()
10 ap.add_argument("-v", "--vgg", required=True,
11 help="path to pre-trained VGGNet for fine-tuning")
12 ap.add_argument("-c", "--checkpoints", required=True,
13 help="path to output checkpoint directory")
14 ap.add_argument("-p", "--prefix", required=True,
15 help="name of model prefix")
16 ap.add_argument("-s", "--start-epoch", type=int, default=0,
17 help="epoch to restart training at")
18 args = vars(ap.parse_args())

20 # set the logging level and output file
21 logging.basicConfig(level=logging.DEBUG,
22 filename="training_{}.log".format(args["start_epoch"]),
23 filemode="w")
24
25 # determine the batch
26 batchSize = config.BATCH_SIZE * config.NUM_DEVICES

28 # construct the training image iterator
29 trainIter = mx.io.ImageRecordIter(
30 path_imgrec=config.TRAIN_MX_REC,
31 data_shape=(3, 224, 224),
32 batch_size=batchSize,
33 rand_crop=True,
34 rand_mirror=True,
35 rotate=15,
36 max_shear_ratio=0.1,
37 mean_r=config.R_MEAN,
38 mean_g=config.G_MEAN,
39 mean_b=config.B_MEAN,
40 preprocess_threads=config.NUM_DEVICES * 2)

42 # construct the validation image iterator
43 valIter = mx.io.ImageRecordIter(
44 path_imgrec=config.VAL_MX_REC,
45 data_shape=(3, 224, 224),
46 batch_size=batchSize,
47 mean_r=config.R_MEAN,
48 mean_g=config.G_MEAN,
49 mean_b=config.B_MEAN)

51 # initialize the optimizer and the training contexts
52 opt = mx.optimizer.SGD(learning_rate=1e-4, momentum=0.9, wd=0.0005,
53 rescale_grad=1.0 / batchSize)
54 ctx = [mx.gpu(3)]

56 # construct the checkpoints path, initialize the model argument and
57 # auxiliary parameters, and whether uninitialized parameters should
58 # be allowed
59 checkpointsPath = os.path.sep.join([args["checkpoints"],
60 args["prefix"]])
61 argParams = None
62 auxParams = None
63 allowMissing = False

65 # if there is no specific model starting epoch supplied, then we
66 # need to build the network architecture
67 if args["start_epoch"] <= 0:
68 # load the pre-trained VGG16 model
69 print("[INFO] loading pre-trained model...")
70 (symbol, argParams, auxParams) = mx.model.load_checkpoint(
71 args["vgg"], 0)
72 allowMissing = True

74 # grab the layers from the pre-trained model, then find the
75 # dropout layer *prior* to the final FC layer (i.e., the layer
76 # that contains the number of class labels)
77 # HINT: you can find layer names like this:
78 # for layer in layers:
79 # print(layer.name)
80 # then, append the string ¡®_output¡® to the layer name
81 layers = symbol.get_internals()
82 net = layers["drop7_output"]

84 # construct a new FC layer using the desired number of output
85 # class labels, followed by a softmax output
86 net = mx.sym.FullyConnected(data=net,
87 num_hidden=config.NUM_CLASSES, name="fc8")
88 net = mx.sym.SoftmaxOutput(data=net, name="softmax")
89
90 # construct a new set of network arguments, removing any previous
91 # arguments pertaining to FC8 (this will allow us to train the
92 # final layer)
93 argParams = dict({k:argParams[k] for k in argParams
94 if "fc8" not in k})

96 # otherwise, a specific checkpoint was supplied
97 else:
98 # load the checkpoint from disk
99 print("[INFO] loading epoch {}...".format(args["start_epoch"]))
100 (net, argParams, auxParams) = mx.model.load_checkpoint(
101 checkpointsPath, args["start_epoch"])

103 # initialize the callbacks and evaluation metrics
104 batchEndCBs = [mx.callback.Speedometer(batchSize, 50)]
105 epochEndCBs = [mx.callback.do_checkpoint(checkpointsPath)]
106 metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5),
107 mx.metric.CrossEntropy()]

109 # construct the model and train it
110 print("[INFO] training network...")
111 model = mx.mod.Module(symbol=net, context=ctx)
112 model.fit(
113 trainIter,
114 eval_data=valIter,
115 num_epoch=65,
116 begin_epoch=args["start_epoch"],
117 initializer=mx.initializer.Xavier(),
118 arg_params=argParams,
119 aux_params=auxParams,
120 optimizer=opt,
121 allow_missing=allowMissing,
122 eval_metric=metrics,
123 batch_end_callback=batchEndCBs,
124 epoch_end_callback=epochEndCBs)