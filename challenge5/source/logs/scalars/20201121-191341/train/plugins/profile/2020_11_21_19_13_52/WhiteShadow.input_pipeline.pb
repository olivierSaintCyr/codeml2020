	�Ü��T@�Ü��T@!�Ü��T@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�Ü��T@��A�f@1ٕ��z�R@A;ŪA�?Iٗl<�b@*	��n���@2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map��I*S�D@!�<�/yT@)͑�_�D@1�>���]T@:Preprocessing2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map��h���@!a�{�[&@)��sC@1g�f�%@:Preprocessing2j
3Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2ݶ�Q�I@!��Չ�X@)�b('�5@1U��&�@:Preprocessing2y
AIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip����6H@!QVv��W@)�������?1�`l�>�?:Preprocessing2t
<Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle��0}�!8H@!��Zk��W@)�d��]��?1����zz�?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map::TensorSlice�U������?!o*�e'W�?)U������?1o*�e'W�?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map::TensorSlice��?�:s�?!t@_�x�?)�?�:s�?1t@_�x�?:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImplS�q��I@!/�n��X@)�d73�р?1�l2U�s�?:Preprocessing2F
Iterator::ModelO!W�Y�?!�����?).�;1�ŀ?1E�h�?:Preprocessing2P
Iterator::Model::Prefetch�w���?!ƕ�)�?)�w���?1ƕ�)�?:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCache����I@!��2 �X@)��s�fl?1ht-��{?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�6.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��A�f@��A�f@!��A�f@      ��!       "	ٕ��z�R@ٕ��z�R@!ٕ��z�R@*      ��!       2	;ŪA�?;ŪA�?!;ŪA�?:	ٗl<�b@ٗl<�b@!ٗl<�b@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb 