�	��|	��`@��|	��`@!��|	��`@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��|	��`@w0b� 
	@1��0Bx0`@A� �В?I�fe���@*A�l�,�@)      �=2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map��WWjBS@!��CK�JT@)L�u�"S@1j��`)T@:Preprocessing2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map���]i	&@!2�Hm7'@)��s�� %@18�8R� &@:Preprocessing2j
3Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2���ɸW@!#Zi@�X@)u�BYx@1e�Ƀz�@:Preprocessing2y
AIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip�k�m�\OV@!���gt�W@)�����?1�}"����?:Preprocessing2t
<Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle��x�0D�V@!Lx��W@)�#ӡ���?1��L�,J�?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map::TensorSlice�}˜.���?!N�dol�?)}˜.���?1N�dol�?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map::TensorSlice�������?!` M���?)������?1` M���?:Preprocessing2F
Iterator::Model�.���?!f��?���?)����k�?1�c���Z�?:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImpl���B�W@!�*/��X@)�0&��~?1�;(t�?:Preprocessing2P
Iterator::Model::Prefetchߌ����}?!��%%?)ߌ����}?1��%%?:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCacheI���p�W@!�����X@)�<֌rg?1 �⦸�h?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	w0b� 
	@w0b� 
	@!w0b� 
	@      ��!       "	��0Bx0`@��0Bx0`@!��0Bx0`@*      ��!       2	� �В?� �В?!� �В?:	�fe���@�fe���@!�fe���@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb �"o
?sequential/random_rotation/transform/ImageProjectiveTransformV2ImageProjectiveTransformV2����?!����?"u
Kgradient_tape/sequential/densenet121/conv1/conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterAmI���?!���W�є?"u
Kgradient_tape/sequential/densenet121/pool2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterE��w�f}?!O#�uU+�?"-
IteratorGetNext/_4_Recv�x>�z�|?!��,z��?"~
Tgradient_tape/sequential/densenet121/conv2_block6_1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter\�(;�{?!�����?"V
0sequential/densenet121/conv1/bn/FusedBatchNormV3FusedBatchNormV3�j��Zm{?!H�l8���?"u
Kgradient_tape/sequential/densenet121/pool3_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter����x?!d��J䍫?"~
Tgradient_tape/sequential/densenet121/conv2_block5_1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��M
)w?!��@�s�?"_
9sequential/densenet121/conv3_block8_0_bn/FusedBatchNormV3FusedBatchNormV3A�����v?!��1��?"l
Bgradient_tape/sequential/densenet121/conv1/bn/FusedBatchNormGradV3FusedBatchNormGradV3��>�v?!=���?I���f�i?Q2/��X@Y���z��T@aM!��1@q#;��ku�?y8DB�Y?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 