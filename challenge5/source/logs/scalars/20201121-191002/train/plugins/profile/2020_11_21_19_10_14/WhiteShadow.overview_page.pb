�	I��&�hT@I��&�hT@!I��&�hT@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-I��&�hT@�-�R2@1��7ہR@Ai�ai�G�?IH�9���@*	�$���@2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map�q���E@!۰��kT@)�6��E@1��@l�OT@:Preprocessing2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map�(�x��b@!8�r��'@)t#,*�@1
|�X�'@:Preprocessing2j
3Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2�����J@!�>�Y��X@)�����@1�Q�nW�@:Preprocessing2y
AIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip�_&���1I@!SV��ɣW@)"��`f�?1��H���?:Preprocessing2t
<Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle���ݒdI@!�����W@)e��)1�?1��9���?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map::TensorSlice����u6��?!`E3p�?)���u6��?1`E3p�?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map::TensorSlice�~V�)���?!�%9c	�?)~V�)���?1�%9c	�?:Preprocessing2F
Iterator::Model����
�?!#M����?)�,��V�?1���e��?:Preprocessing2P
Iterator::Model::Prefetchl���f}?!���{J��?)l���f}?1���{J��?:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImpl�&N�w�J@!�M���X@)0� ���{?1P�D�,/�?:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCache=Fy��J@!�6�.�X@)���
~k?1��t���y?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�-�R2@�-�R2@!�-�R2@      ��!       "	��7ہR@��7ہR@!��7ہR@*      ��!       2	i�ai�G�?i�ai�G�?!i�ai�G�?:	H�9���@H�9���@!H�9���@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb �"o
?sequential/random_rotation/transform/ImageProjectiveTransformV2ImageProjectiveTransformV2n�E�\�?!n�E�\�?"M
1sequential/densenet121/conv2_block3_2_conv/Conv2DConv2D&��?!���N��?"u
Kgradient_tape/sequential/densenet121/conv1/conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter1�^oo��?!P�Y�Q�?"-
IteratorGetNext/_2_Recv��Ɗ5��?!��e�̢?"u
Kgradient_tape/sequential/densenet121/pool2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�hƴ��}?!,`w����?"~
Tgradient_tape/sequential/densenet121/conv2_block6_1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�q�B�V{?!l��df�?"u
Kgradient_tape/sequential/densenet121/pool3_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter)A>�w?!���,n�?"~
Tgradient_tape/sequential/densenet121/conv2_block5_1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterv�\i�v?!�PrX�¯?"`
:sequential/densenet121/conv4_block13_1_bn/FusedBatchNormV3FusedBatchNormV3 ��g�t?!.�b-�?"l
Bgradient_tape/sequential/densenet121/pool2_bn/FusedBatchNormGradV3FusedBatchNormGradV3n.��t?!��F��w�?I�3R^��v?Q��Y��X@YpC�r�R@a?�5�;@q���Q���?y����9�d?"�

device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 