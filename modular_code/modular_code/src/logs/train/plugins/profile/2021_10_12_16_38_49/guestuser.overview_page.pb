�	���~��@���~��@!���~��@	�
[�0?�
[�0?!�
[�0?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$���~��@N(D�!T�?A��C���@Y��d��J�?*	J+�&Q@2F
Iterator::Model�h�����?!�A~��C@)[}uU��?1|�^�4,;@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat��"[A�?!m%2i;@)��:��?1��y�,9@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�{�E{��?!�z�t
�7@)���/�?1_���2@:Preprocessing2S
Iterator::Model::ParallelMap|�5Z�?!�;�&"(@)|�5Z�?1�;�&"(@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�,�cyW�?!!���[aN@)���<,t?1l2̠c�@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice1'h��'m?!�y��g�@)1'h��'m?1�y��g�@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapy\T��b�?!;J���+:@)�	.V�`Z?1�z��c�@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor]�@�"Y?![��E��@)]�@�"Y?1[��E��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	N(D�!T�?N(D�!T�?!N(D�!T�?      ��!       "      ��!       *      ��!       2	��C���@��C���@!��C���@:      ��!       B      ��!       J	��d��J�?��d��J�?!��d��J�?R      ��!       Z	��d��J�?��d��J�?!��d��J�?JCPU_ONLY2black"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 