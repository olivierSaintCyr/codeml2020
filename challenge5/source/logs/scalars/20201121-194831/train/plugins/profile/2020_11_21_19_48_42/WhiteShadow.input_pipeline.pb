	ԁ��VU@ԁ��VU@!ԁ��VU@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ԁ��VU@f`X~@1ݔ�Z	�R@A�dT�ݐ?I��{���@*	-���E�@2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map�d��S�F@!1�2YST@)s֧��F@1pl/�5T@:Preprocessing2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map��%VF#�@!HE�^��(@)k�MG�@1����'@:Preprocessing2j
3Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2o�KS�K@!J �8�X@)}?5^�	@1���@:Preprocessing2y
AIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip�LKuJ@!ю��W@)���a�2�?1�=�<�/�?:Preprocessing2t
<Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle�w�h�h�J@!.a�P�W@)�4-�2�?1Y.�J�s�?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map::TensorSlice����	�_�?!*/]�)Q�?)���	�_�?1*/]�)Q�?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map::TensorSlice�*T7��?!���
���?)*T7��?1���
���?:Preprocessing2F
Iterator::Model��J
,�?!j���4��?),��̰�?1��ũ:��?:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImpl�_?��K@!~S���X@)�'�8'�?1bf�I�?:Preprocessing2P
Iterator::Model::Prefetch����N}?!3k��.<�?)����N}?13k��.<�?:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCache�CV�K@!W��0�X@)��հ�c?1XKv�l�q?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�8.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	f`X~@f`X~@!f`X~@      ��!       "	ݔ�Z	�R@ݔ�Z	�R@!ݔ�Z	�R@*      ��!       2	�dT�ݐ?�dT�ݐ?!�dT�ݐ?:	��{���@��{���@!��{���@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb 