�	�����V@�����V@!�����V@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�����V@����@1��:qT@A`u�Hg`�?Iy�t�@*	�&1h	�@2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map��W�B�F@!�t��WT@){2��fF@1�׳�S:T@:Preprocessing2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map����kq@!{�|��(@)ĵ��^�@1<Y{�'@:Preprocessing2j
3Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2)$��;�K@!=P
K�X@)�+��!@1����.@:Preprocessing2y
AIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip���c9J@!��֭W@)��5���?1���	Q��?:Preprocessing2t
<Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle�o�jJ@!�f޻'�W@)N�@�C��?1�����(�?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map::TensorSlice�`�U,~S�?!+9D� |�?)`�U,~S�?1+9D� |�?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map::TensorSlice������!�?!�G~�N�?)�����!�?1�G~�N�?:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImplgF?N�K@! '����X@)��#bJ$�?1*U��6��?:Preprocessing2F
Iterator::ModelZ�N��?!A���$�?)���{h?1M��\�?:Preprocessing2P
Iterator::Model::Prefetch�S�<z?!��W<�?)�S�<z?1��W<�?:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCache�x[鵭K@!����]�X@)vk���i?1��vgMow?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����@����@!����@      ��!       "	��:qT@��:qT@!��:qT@*      ��!       2	`u�Hg`�?`u�Hg`�?!`u�Hg`�?:	y�t�@y�t�@!y�t�@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb �"D
(sequential/densenet121/pool4_conv/Conv2DConv2D��©X�?!��©X�?"`
:sequential/densenet121/conv4_block20_0_bn/FusedBatchNormV3FusedBatchNormV3YH�!��?!�t�����?"~
Tgradient_tape/sequential/densenet121/conv2_block5_1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�uȄ?!j�_R��?"o
?sequential/random_rotation/transform/ImageProjectiveTransformV2ImageProjectiveTransformV2��2�,{�?!毬MЧ?"|
Sgradient_tape/sequential/densenet121/conv3_block6_2_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputh�]�.׃?!����Ŭ?"_
Agradient_tape/sequential/densenet121/conv3_block8_1_relu/ReluGradReluGrad�qG\�?!5?P�İ?"N
2sequential/densenet121/conv4_block13_1_conv/Conv2DConv2D[�/�e�?! 8���?"u
Kgradient_tape/sequential/densenet121/conv1/conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteru=8��H�?!�?]��?"l
Bgradient_tape/sequential/densenet121/pool2_bn/FusedBatchNormGradV3FusedBatchNormGradV38B�+�?!҃�8B�?"u
Kgradient_tape/sequential/densenet121/pool2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterL�.s<r{?!7p fѸ?I��&3Yu{?Qd3�*��X@Y˷|˷R@a� � �;@q�����?yt��?�Ee?"�

device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 