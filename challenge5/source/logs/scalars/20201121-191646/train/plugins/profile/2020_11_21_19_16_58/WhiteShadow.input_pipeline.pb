	'��UT@'��UT@!'��UT@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-'��UT@P�s'�@1��f�R�R@A���Fu:�?I-��o��@*	�� ���@2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map�1}�!8E@!h�.a��T@)8�ܘ��D@1�|%sT@:Preprocessing2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map��c��@!�Gq��%@)e�z�F�@1Oc�H�%@:Preprocessing2j
3Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2sJ_�I@!<����X@)
If�w@1���j�@:Preprocessing2y
AIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip���̰1H@!��]��W@)�	�ʼU�?1��ߦ���?:Preprocessing2t
<Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle���o�_H@!��^��W@)}ԛQ��?1�g�5�Z�?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map::TensorSlice����҈��?!�W ���?)���҈��?1�W ���?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map::TensorSlice��Z��8�?!d�W�v�?)�Z��8�?1d�W�v�?:Preprocessing2F
Iterator::Model��t=�u�?!�����?)�����S�?1��űOӒ?:Preprocessing2P
Iterator::Model::Prefetch�.ޏ�/?!�"���`�?)�.ޏ�/?1�"���`�?:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImpl�����I@!�:��X@)!v��y�}?1Ѫ�9&Ɍ?:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCache�X S�I@!������X@)Zd;�O�g?1���s��v?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�5.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	P�s'�@P�s'�@!P�s'�@      ��!       "	��f�R�R@��f�R�R@!��f�R�R@*      ��!       2	���Fu:�?���Fu:�?!���Fu:�?:	-��o��@-��o��@!-��o��@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb 