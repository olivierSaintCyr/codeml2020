�	���BU@���BU@!���BU@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���BU@��+I��@1s���6�S@A�I�p�?I��1Z@*	��K7!��@2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map�l=C8f�E@!��v��XT@)R�.��tE@1Lfo�9T@:Preprocessing2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map��}��@!���Pk''@)�;��@1��A6&@:Preprocessing2j
3Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2��A$�J@!�����X@)�����@1�ƾ	|@:Preprocessing2y
AIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip�ڍ>��H@!@_�LąW@)���2�'�?1�j*V��?:Preprocessing2t
<Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle�G��t%I@!@� B_�W@)�6ǹM��?1��zM�?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map::TensorSlice�(5
If�?!���*��?)(5
If�?1���*��?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map::TensorSlice�V*����?!�&�I%�?)V*����?1�&�I%�?:Preprocessing2F
Iterator::Model��^D�1�?!�����?)� 3��O�?1��y7��?:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImpl�]=�J@!���(�X@)�R��%��?1n�B���?:Preprocessing2P
Iterator::Model::Prefetch�PS�'|?!%�Dç��?)�PS�'|?1%�Dç��?:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCache{fI���J@!�KJ���X@)�BY��Zg?1�M��.v?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�4.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��+I��@��+I��@!��+I��@      ��!       "	s���6�S@s���6�S@!s���6�S@*      ��!       2	�I�p�?�I�p�?!�I�p�?:	��1Z@��1Z@!��1Z@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb �"M
1sequential/densenet121/conv2_block3_2_conv/Conv2DConv2D��|?�?!��|?�?"~
Tgradient_tape/sequential/densenet121/conv2_block5_1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��5l��?!����U��?"o
?sequential/random_rotation/transform/ImageProjectiveTransformV2ImageProjectiveTransformV2�����W�?!��4Nt�?"u
Kgradient_tape/sequential/densenet121/conv2_block4_1_bn/FusedBatchNormGradV3FusedBatchNormGradV3�Q!��?!�hv����?"u
Kgradient_tape/sequential/densenet121/conv1/conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��A���?!����+�?"@
&sequential/densenet121/pool2_relu/ReluRelu&'1�?!w��
�"�?"D
(sequential/densenet121/pool4_conv/Conv2DConv2D�U�7�}?!�C ��o�?"v
Lgradient_tape/sequential/densenet121/conv3_block12_0_bn/FusedBatchNormGradV3FusedBatchNormGradV3E~��ɝ}?!}[���I�?"_
9sequential/densenet121/conv2_block1_1_bn/FusedBatchNormV3FusedBatchNormV3���}?!��T���?"u
Kgradient_tape/sequential/densenet121/pool2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterz�F�"|?!}����ܵ?I��6�}w?Q/%g��X@Y����R@a���;@q�[0���?y�b��f?"�

both�Your program is POTENTIALLY input-bound because 3.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�4.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 