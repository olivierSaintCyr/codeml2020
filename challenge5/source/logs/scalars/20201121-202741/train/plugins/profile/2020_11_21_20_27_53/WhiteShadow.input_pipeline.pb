	��\�T@��\�T@!��\�T@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��\�T@"nN% �?1��TO�R@A�~31]��?I$&��[�@*	1�&
�@2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map�T�d�_G@!8	.ZT@)�p�GRBG@1@r"uj@T@:Preprocessing2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map���6��@!�х�)@)�ڥ��@1/��'J(@:Preprocessing2j
3Iterator::Model::Prefetch::MemoryCacheImpl::BatchV27�xͫ�L@!�O���X@)��� ��@1>*ǉ)@:Preprocessing2y
AIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip���n�5K@!�LK�&�W@)n/��?1�L�Ûr�?:Preprocessing2t
<Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle�jM�bK@!���k�W@)D�+g��?1Ѹ��{��?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map::TensorSlice�/�o��e�?!|��渘�?)/�o��e�?1|��渘�?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map::TensorSlice�N�#Ed�?!;�ī��?)N�#Ed�?1;�ī��?:Preprocessing2F
Iterator::ModelZ�b+hZ�?!Q�?����?)^-wf��?1�����%�?:Preprocessing2P
Iterator::Model::PrefetchUm7�7M?!Q5�A�?)Um7�7M?1Q5�A�?:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImplp{��v�L@!s��N��X@)��6�^y?1��64�?:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCache9�ֱL@!x�F�X@)�R����g?1Hf���t?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	"nN% �?"nN% �?!"nN% �?      ��!       "	��TO�R@��TO�R@!��TO�R@*      ��!       2	�~31]��?�~31]��?!�~31]��?:	$&��[�@$&��[�@!$&��[�@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb 