	F����=T@F����=T@!F����=T@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-F����=T@̙�
}0@1�����R@A�_���?IZI+��@*	z�GA��@2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map���x���D@!s�\,��T@)�聏�D@1�sO*{T@:Preprocessing2�
IIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map�狽_�@!l8�Uid%@) $���@1�d�p�$@:Preprocessing2j
3Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2s��YI@!"����X@)[(���9@1��QW�@:Preprocessing2y
AIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip����H��G@!(2����W@)����?1�)��K��?:Preprocessing2t
<Iterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle���iTH@!��ǎ��W@)3O�)���?1��<~�?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map::TensorSlice����&�?!@�*��?)���&�?1@�*��?:Preprocessing2�
VIterator::Model::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::Map::TensorSlice���L����?!�{���i�?)��L����?1�{���i�?:Preprocessing2F
Iterator::Model�2p@KW�?!2L���?)\t��z��?1%S �~�?:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImpl�`��ZI@!�B���X@)�TގpZ�?1W���1�?:Preprocessing2P
Iterator::Model::Prefetch��[X7�}?!��x�.p�?)��[X7�}?1��x�.p�?:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCache�T�	g[I@!7$���X@)&��|�k?1c�^x�m{?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�5.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	̙�
}0@̙�
}0@!̙�
}0@      ��!       "	�����R@�����R@!�����R@*      ��!       2	�_���?�_���?!�_���?:	ZI+��@ZI+��@!ZI+��@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb 