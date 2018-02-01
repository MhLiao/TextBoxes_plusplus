from __future__ import print_function
caffe_root = './'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
import caffe
from caffe.model_libs import *
from google.protobuf import text_format
import modelConfig
from modelConfig import AddExtraLayers, config

import math
import shutil
import stat
import subprocess

caffe_root = os.getcwd()

# Create train net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(config['train_data'], batch_size=modelConfig.batch_size_per_device,
        train=True, output_label=True, label_map_file=config['label_map_file'],
        transform_param=modelConfig.train_transform_param, batch_sampler=modelConfig.batch_sampler)

VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
    dropout=False, freeze_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2'])

AddExtraLayers(net, False, lr_mult=config['lr_mult'])

mbox_layers = CreateMultiBoxHead_multitask(net, data_layer='data', from_layers=modelConfig.mbox_source_layers,
        use_batchnorm=False, min_sizes=modelConfig.min_sizes, max_sizes=modelConfig.max_sizes, use_polygon=config['use_polygon'],
        aspect_ratios=modelConfig.aspect_ratios, steps=modelConfig.steps, normalizations=modelConfig.normalizations,
        num_classes=modelConfig.num_classes, share_location=modelConfig.share_location, flip=config['flip'], clip=config['clip'],
        prior_variance=modelConfig.prior_variance, denser_prior_boxes=config['denser_prior_boxes'], kernel_size=[3,5], pad=[1,2], lr_mult=config['lr_mult'])

# Create the MultiBoxLossLayer.
name = "mbox_loss"
mbox_layers.append(net.label)
net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=modelConfig.multibox_loss_param,
        loss_param=modelConfig.loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
        propagate_down=[True, True, False, False])

conf_name = "mbox_conf"
with open(modelConfig.train_net_file, 'w') as f:
    print('name: "{}_train"'.format(modelConfig.model_name), file=f)
    print(net.to_proto(), file=f)
shutil.copy(modelConfig.train_net_file, modelConfig.job_dir)

net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(config['test_data'], batch_size=modelConfig.test_batch_size,
        train=False, output_label=True, label_map_file=config['label_map_file'],
        transform_param=modelConfig.test_transform_param)

VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
    dropout=False)

AddExtraLayers(net, False, lr_mult=config['lr_mult'])

mbox_layers = CreateMultiBoxHead_multitask(net, data_layer='data', from_layers=modelConfig.mbox_source_layers,
        use_batchnorm=False, min_sizes=modelConfig.min_sizes, max_sizes=modelConfig.max_sizes, use_polygon=config['use_polygon'],
        aspect_ratios=modelConfig.aspect_ratios, steps=modelConfig.steps, normalizations=modelConfig.normalizations,
        num_classes=modelConfig.num_classes, share_location=modelConfig.share_location, flip=config['flip'], clip=config['clip'],
        prior_variance=modelConfig.prior_variance, denser_prior_boxes=config['denser_prior_boxes'], kernel_size=[3,5], pad=[1,2], lr_mult=config['lr_mult'])

conf_name = "mbox_conf"
reshape_name = "{}_reshape".format(conf_name)
net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, modelConfig.num_classes]))
softmax_name = "{}_softmax".format(conf_name)
net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
flatten_name = "{}_flatten".format(conf_name)
net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
mbox_layers[1] = net[flatten_name]

net.detection_out = L.DetectionOutput(*mbox_layers,
    detection_output_param=modelConfig.det_out_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
    detection_evaluate_param=modelConfig.det_eval_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))

with open(modelConfig.test_net_file, 'w') as f:
    print('name: "{}_test"'.format(modelConfig.model_name), file=f)
    print(net.to_proto(), file=f) 
shutil.copy(modelConfig.test_net_file, modelConfig.job_dir)

# Create deploy net.
# Remove the first and last layer from test net.
deploy_net = net
with open(modelConfig.deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
    net_param.name = '{}_deploy'.format(modelConfig.model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, config['resize_height'], config['resize_width']])])
    print(net_param, file=f)
shutil.copy(modelConfig.deploy_net_file, modelConfig.job_dir)

# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=modelConfig.train_net_file,
        test_net=[modelConfig.test_net_file],
        snapshot_prefix=modelConfig.snapshot_prefix,
        **modelConfig.solver_param)

with open(modelConfig.solver_file, 'w') as f:
    print(solver, file=f)
shutil.copy(modelConfig.solver_file, modelConfig.job_dir)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(modelConfig.snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(modelConfig.model_name))[1])
    if iter > max_iter:
      max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(config['pretrain_model'])
if config['resume_training']:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(modelConfig.snapshot_prefix, max_iter)

if config['remove_old_models']:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(modelConfig.snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(modelConfig.model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(modelConfig.snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(modelConfig.model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(modelConfig.snapshot_dir, file))

# Create job file.
with open(modelConfig.job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(modelConfig.solver_file))
  f.write(train_src_param)
  if modelConfig.solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(modelConfig.gpus, modelConfig.job_dir, modelConfig.model_name))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(modelConfig.job_dir, modelConfig.model_name))

# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, modelConfig.job_dir)

# Run the job.
os.chmod(modelConfig.job_file, stat.S_IRWXU)
if config['run_soon']:
  subprocess.call(modelConfig.job_file, shell=True)
