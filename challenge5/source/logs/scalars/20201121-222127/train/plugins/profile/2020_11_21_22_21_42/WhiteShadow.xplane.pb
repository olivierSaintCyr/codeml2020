
��	/host:CPU������
"
"
"
"
"
 �������" " "
	 ������"
  "  "�º��ȋ�"m�����"
	 ������"
  "  ":9��֦����	"��" *$$1"!*size:4 dest:0 async:1":9������Й"��" *$$1"!*size:4 dest:0 async:1"�����Ȏ�"
	 ������"*
LogicalAnd"
  "
*output" 
"*[]"�  ".��ܗ�����"��" *$$1""8�"�  ":9���׆���"��" *$$1"!*size:1 dest:0 async:1"ȡ����ٯ
" ��ל��Ƴ"�������"
"�˪����"mн��"
	 ������"
  "  "
	 ������"
  "  "n��ʋm"
"
	 ������"
  "  "n�и�n"
	 �Ѕ���"
  "  ʳY����
 �������"  " "
	 �����"*SameWorkerRecvDone"
  "*dynamic" "*[200,10]"�  "D4蘼�����"��" *$$1"!*size:8000 dest:0 async:1"�  "�����"
	 ��ɫ��"*SameWorkerRecvDone"
  "*dynamic" "*
	 �Ԙ���";*7sequential/random_contrast/random_uniform/RandomUniform"
 �������"
*output" "*[]"�  "+5������"ɾ" *$$1""8#"�  "�ؕ��"*gpu_host_bfc"  " ��"	 �����"  " ��" " �"
	 ������"*
 �������"
*output" "*[]"�  "m����"*gpu_host_bfc"  " ��"	 �����"  " ��" " �"
	 ������"
  "  "�  "+6�ٶ����"ʾ" *$$1""8$"�  "����"
	 �ؘ���"*Adam/add"
 �������"
*output" 	"*[]"�  "+7��ǟ��"˾" *$$1""8%"�  "+8������"̾" *$$1""8&"�  "��ѭ"*gpu_host_bfc"  " ��"	 �����"  " ��" " �"
	 ������"*SameWorkerRecvDone"
  "*dynamic" 	"*[]"�  "@9����ǃ"Ͼ" *$$1"!*size:8 dest:0 async:1"�  "�����"
	 �ژ���"*SameWorkerRecvDone"
  "*dynamic" "*[2]"�  "@4���Ы�"Ծ" *$$1"!*size:8 dest:0 async:1"�  "+:�����ʩ"׾" *$$1""8'"�  "x���"
	 �����"
  "  "�  "�����"
	 ������"
 �������"
*output" "*[]"�  "+;�������"ؾ" *$$1""8("�  "s����"
	 �ؘ���"
  "  "�  "+:����͊"پ" *$$1""8)"�  "�����"
	 �ؘ���"*SameWorkerRecvDone"
  "*dynamic" "*[]"�  "@4�������"ܾ" *$$1"!*size:4 dest:0 async:1"�  "+6�������"߾" *$$1""8*"�  "�����"
	 ������"*Adam/Pow"
 �������"
*output" "*[]"�  "+<����辬"�" *$$1""8+"�  "+<谠����"�" *$$1""8,"�  "���ؖ"
	 �����";*7sequential/random_flip/random_flip_left_right/ReverseV2"
 �������"
*output" "*
	 ������"N*Jsequential/random_flip/random_flip_left_right/random_uniform/RandomUniform"
 �������"
*output" "	*[200]"�  "+5�������"�" *$$1""8."�  "+>�������"�" *$$1""8/"�  "+?������"�" *$$1""80"�  "+@������"�" *$$1""81"�  "+A�������"�" *$$1""82"�  "t����"
	 ������"
  "  "�  "+:ج�����"�" *$$1""83"�  "x���"
	 ��ɫ��"
  "  "�  "����"
	 ��ɫ��"8*4sequential/random_flip/random_flip_up_down/ReverseV2"
 �������"
*output" "*
	 ������"K*Gsequential/random_flip/random_flip_up_down/random_uniform/RandomUniform"
 �������"
*output" "	*[200]"�  "+5�������"�" *$$1""85"�  "+>賣����"�" *$$1""86"�  "+?����"��" *$$1""87"�  "+@���ȵ�"�" *$$1""88"�  "+?ࠟ�Ȼ�"�" *$$1""89"�  "tȪ��"
	 ������"
  "  "�  "+:�����Ղ"�" *$$1""8:"�  "x����"
	 �����"
  "  "�  "�����"
	 �����".**sequential/random_contrast/adjust_contrast"
 �������"
*output" "*
	 ������".**sequential/random_contrast/adjust_contrast"
 �������"  "�  "+Bؔ���Ȣ"�" *$$1""8;"�  "+CȌ���Ğ"�" *$$1""8<"�  "+D��لȪ�"�" *$$1""8="�  "+E����Ƶ"��" *$$1""8>"�  "xȃ��"
	 ��ɫ��"
  "  "�  "s��"
	 �Ԙ���"
  "  "�  "�����"*gpu_host_bfc"  " ��"	 �����"  " ��" " �"
	 ������".**sequential/random_rotation/strided_slice_2"
 �������"
*output" "*[]"�  "�Ȝ�"*gpu_host_bfc"  " ��"	 �����"  " ��" " �"
	 ������".**sequential/random_rotation/strided_slice_1"
 �������"
*output" "*[]"�  "�����"
	 �Ԙ���"*SameWorkerRecvDone"
  "*dynamic" "*[]"�  "@4���ث�"��" *$$1"!*size:4 dest:0 async:1"�  "�����"
	 �ؘ���"*SameWorkerRecvDone"
  "*dynamic" "*[]"�  "@4�Ԯ��ϵ"��" *$$1"!*size:4 dest:0 async:1"�  "�����"
	 ������"?*;sequential/random_rotation/stateful_uniform/StatefulUniform"
 �������"
*output" "	*[200]"�  "@4�������"��" *$$1"!*size:4 dest:0 async:0"�  "+F�������"��" *$$1""8?"�  "m���"*gpu_host_bfc"  " ��"	 �����"  " ��" " �"
	 ������"
  "  "�  "+6������"��" *$$1""8@"�  "+G����ކ"��" *$$1""8A"�  "����"
	 ������"4*0sequential/random_rotation/rotation_matrix/Sin_2"
 �������"
*output" "	*[200]"�  "+H�����İ"��" *$$1""8B"�  "��醒"
	 ������"4*0sequential/random_rotation/rotation_matrix/Cos_2"
 �������"
*output" "	*[200]"�  "+Iȓ��軳"��" *$$1""8C"�  "t�ᒜ"
	 ������"
  "  "�  "����"
	 ������"4*0sequential/random_rotation/rotation_matrix/mul_1"
 �������"
*output" "	*[200]"�  "+6�������"��" *$$1""8D"�  "��ι�"
	 ������"4*0sequential/random_rotation/rotation_matrix/mul_2"
 �������"
*output" "	*[200]"�  "+6ؽ���ү"��" *$$1""8E"�  "��Ó�"
	 �����"2*.sequential/random_rotation/rotation_matrix/mul"
 �������"
*output" "	*[200]"�  "+6��ŭ��"��" *$$1""8F"�  "��㧴"
	 �����"4*0sequential/random_rotation/rotation_matrix/mul_3"
 �������"
*output" "	*[200]"�  "+6���ȃ�"��" *$$1""8G"�  "�����"
	 �����"2*.sequential/random_rotation/rotation_matrix/Neg"
 �������"
*output" "*[200,1]"�  "+J��޽���"��" *$$1""8H"�  "+KȜ�����"��" *$$1""8I"�  "t����"
	 ������"
  "  "�  "+:ȶ����"��" *$$1""8J"�  "t���"
	 �����"
  "  "�  "+@؟����"��" *$$1""8K"�  "s����"
	 �ā���"
  "  "�  "+@������"��" *$$1""8L"�  "s�ݩ�"
	 �Ԙ���"
  "  "�  "�����"
	 ������"4*0sequential/random_rotation/rotation_matrix/zeros"
 �������"
*output" "*[200,2]"�  "+L�������"��" *$$1""8M"�  "+M�������"��" *$$1""8N"�  "+M�������"��" *$$1""8O"�  "����"
	 �����"5*1sequential/random_rotation/rotation_matrix/concat"
 �������"
*output" "*[200,8]"�  "+N�ˁ����"��" *$$1""8P"�  "+NЌ���Ͷ"��" *$$1""8Q"�  "+N�������"��" *$$1""8R"�  "+N��ތ���"��" *$$1""8S"�  "+N�懏���"��" *$$1""8T"�  "+N��ޒ���"��" *$$1""8U"�  "+N�϶��Ĭ"��" *$$1""8V"�  "t؎��"
	 �����"
  "  "�  "t�֙"
	 �����"
  "  "�  "t����"
	 ������"
  "  "�  "t薺�"
	 ������"
  "  "�  "t���"
	 ������"
  "  "�  "t����"
	 ������"
  "  "�  "�����"
	 ��ɫ��"C*?sequential/random_rotation/transform/ImageProjectiveTransformV2"
 �������"
*output" "*
	 �����"
  "  "�  "t����"
	 �����"
  "  "�  "���ө"
	 ������"-*)sequential/densenet169/zero_padding2d/Pad"
 �������"
*output" "*
	 ��ɫ��"
  "  "�  "��ힲ"
	 ������"u*qgradient_tape/sequential/densenet169/conv1/conv/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer"
 �������"
*output" "*
	 ������"
  "  "�  "�����"
	 ������",*(sequential/densenet169/conv1/conv/Conv2D"
 �������"
*output" "*[200,64,16,16]"�  "���ֽ"
	 ������",*(sequential/densenet169/conv1/conv/Conv2D"
 �������"*temp" "*
[64,3,7,7]"�  "+R��ҿ���"��" *$$1""8Z"�  "�����"
	 ������",*(sequential/densenet169/conv1/conv/Conv2D"
 �������"*temp" "
*[1544]"�  "+S�����˟"��" *$$1""8["�  "+T�ư�迎"��" *$$1""8\"�  "�����"
	 ������",*(sequential/densenet169/conv1/conv/Conv2D"
 �������"  "�  "�����"
	 ������",*(sequential/densenet169/conv1/conv/Conv2D"
 �������"  "�  "����"
	 ������"4*0sequential/densenet169/conv1/bn/FusedBatchNormV3"
 �������"
*output" "*[200,64,16,16]"�  "�����"
	 �Ԙ���"4*0sequential/densenet169/conv1/bn/FusedBatchNormV3"
 �������"
*output" "*[64]"�  "�����"
	 �ؘ���"4*0sequential/densenet169/conv1/bn/FusedBatchNormV3"
 �������"
*output" "*[64]"�  "�����"
	 �ā���"4*0sequential/densenet169/conv1/bn/FusedBatchNormV3"
 �������"
*output" "*[64]"�  "�����"
	 ������"4*0sequential/densenet169/conv1/bn/FusedBatchNormV3"
 �������"
*output" "*[64]"�  "BU������"��" *$$1"!*size:256 dest:0 async:1"�  "BU�������"��" *$$1"!*size:256 dest:0 async:1"�  "+V�������"��" *$$1""8]"�  "+W�����"��" *$$1""8^"�  "�����"
	 ������"2*.sequential/densenet169/conv1/bn/AssignNewValue"
 �������"  "�  "���Ȋ"
	 ������"4*0sequential/densenet169/conv1/bn/AssignNewValue_1"
 �������"  "�  "���"
	 ������"/*+sequential/densenet169/zero_padding2d_1/Pad"
 �������"*temp" "*
	 ������"(*$sequential/densenet169/pool1/MaxPool"
 �������"
*output" "*[200,64,8,8]"�  "+YІ����"��" *$$1""8`"�  "��ⴢ"
	 ������"=*9sequential/densenet169/conv2_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,64,8,8]"�  "���"
	 ����"=*9sequential/densenet169/conv2_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "*[64]"�  "���ܣ"
	 ������"=*9sequential/densenet169/conv2_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "*[64]"�  "���ܤ"
	 ������"=*9sequential/densenet169/conv2_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "*[64]"�  "��Ε�"
	 ������"=*9sequential/densenet169/conv2_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "*[64]"�  "BU�������"��" *$$1"!*size:256 dest:0 async:1"�  "BU�ɜ����"��" *$$1"!*size:256 dest:0 async:1"�  "+V�ń���"��" *$$1""8a"�  "+W�䤵��"��" *$$1""8b"�  "�Ђ��"
	 ������";*7sequential/densenet169/conv2_block1_0_bn/AssignNewValue"
 �������"  "�  "��ʨ�"
	 ������"=*9sequential/densenet169/conv2_block1_0_bn/AssignNewValue_1"
 �������"  "�  "���"
	 ������"5*1sequential/densenet169/conv2_block1_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv2_block1_1_conv/Conv2D"
 �������"*temp" "*[128,64,1,1]"�  "+R������"��" *$$1""8c"�  "�����"
	 ������"5*1sequential/densenet169/conv2_block1_1_conv/Conv2D"
 �������"*temp" "	*[392]"�  "+S�������"¿" *$$1""8d"�  "+Z�Ҙ���"ſ" *$$1""8e"�  "�����"
	 ������"5*1sequential/densenet169/conv2_block1_1_conv/Conv2D"
 �������"  "�  "�����"
	 ������"5*1sequential/densenet169/conv2_block1_1_conv/Conv2D"
 �������"  "�  "�����"
3�?" ����" ���" ���"
	 ������"=*9sequential/densenet169/conv2_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "*
3�?" ����" �" �"
	 ������"=*9sequential/densenet169/conv2_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����"
3�?" ����" �" �"
	 ������"=*9sequential/densenet169/conv2_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����"
3�?" ����" �" �"
	 ������"=*9sequential/densenet169/conv2_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�ж��"
3�?" ����" �" �"
	 ������"=*9sequential/densenet169/conv2_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�������"ǿ" *$$1"!*size:512 dest:0 async:1"�  "BU�ƅ�Е�"ȿ" *$$1"!*size:512 dest:0 async:1"�  "+V����ȧ�"ʿ" *$$1""8f"�  "+W�����Р"̿" *$$1""8g"�  "�����"
3�?" ����" �" �"
	 ������";*7sequential/densenet169/conv2_block1_1_bn/AssignNewValue"
 �������"  "�  "�����"
3�?" ����" �" �"
	 ������"=*9sequential/densenet169/conv2_block1_1_bn/AssignNewValue_1"
 �������"  "�  "���ƀ"
	 ������"5*1sequential/densenet169/conv2_block1_2_conv/Conv2D"
 �������"
*output" "*[200,32,8,8]"�  "�����"
	 ��؄��"5*1sequential/densenet169/conv2_block1_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  "+R��у�͔"Ϳ" *$$1""8h"�  "�����"
	 ������"5*1sequential/densenet169/conv2_block1_2_conv/Conv2D"
 �������"*temp" "*
[19759104]"�  "+[�͕����"п" *$$1""8i"�  "+\�������"ѿ" *$$1""8j"�  "+]��Ù��"ӿ" *$$1""8k"�  "+^Щ�����"տ" *$$1""8l"�  "�����"
	 ������"5*1sequential/densenet169/conv2_block1_2_conv/Conv2D"
 �������"  "�  "��ɐ�"
	 ��؄��"5*1sequential/densenet169/conv2_block1_2_conv/Conv2D"
 �������"  "�  "����"
	 ������"5*1sequential/densenet169/conv2_block1_concat/concat"
 �������"
*output" "*[200,96,8,8]"�  "+N��Ϭ���"׿" *$$1""8m"�  "+N����ް"ؿ" *$$1""8n"�  "v�؇�"
	 ������"
  "  "�  "�����"
	 ������"=*9sequential/densenet169/conv2_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,96,8,8]"�  "���˺"
	 ������"=*9sequential/densenet169/conv2_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "*[96]"�  "�����"
	 ������"=*9sequential/densenet169/conv2_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "*[96]"�  "��濻"
	 ������"=*9sequential/densenet169/conv2_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "*[96]"�  "�К��"
	 ������"=*9sequential/densenet169/conv2_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "*[96]"�  "BU��Ž��"ٿ" *$$1"!*size:384 dest:0 async:1"�  "BU�Ǭ���"ڿ" *$$1"!*size:384 dest:0 async:1"�  "+V�Î����"ܿ" *$$1""8o"�  "+W�������"޿" *$$1""8p"�  "��΄�"
	 ������";*7sequential/densenet169/conv2_block2_0_bn/AssignNewValue"
 �������"  "�  "����"
	 ������"=*9sequential/densenet169/conv2_block2_0_bn/AssignNewValue_1"
 �������"  "�  "�ص��"
	 ������"5*1sequential/densenet169/conv2_block2_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv2_block2_1_conv/Conv2D"
 �������"*temp" "*[128,96,1,1]"�  "+R�������"߿" *$$1""8q"�  "�ȝ��"
	 ������"5*1sequential/densenet169/conv2_block2_1_conv/Conv2D"
 �������"*temp" "	*[392]"�  "+S������"�" *$$1""8r"�  "+Z�������"�" *$$1""8s"�  "�����"
	 ������"5*1sequential/densenet169/conv2_block2_1_conv/Conv2D"
 �������"  "�  "��Џ�"
	 ������"5*1sequential/densenet169/conv2_block2_1_conv/Conv2D"
 �������"  "�  "����"
	 ������"=*9sequential/densenet169/conv2_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv2_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����"
	 �����"=*9sequential/densenet169/conv2_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��̵�"
	 ������"=*9sequential/densenet169/conv2_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���"
	 �����"=*9sequential/densenet169/conv2_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU��چ���"�" *$$1"!*size:512 dest:0 async:1"�  "BU������"�" *$$1"!*size:512 dest:0 async:1"�  "+V������"�" *$$1""8t"�  "+W�������"�" *$$1""8u"�  "�����"
	 �Ā���";*7sequential/densenet169/conv2_block2_1_bn/AssignNewValue"
 �������"  "�  "�����"
	 �Ȁ���"=*9sequential/densenet169/conv2_block2_1_bn/AssignNewValue_1"
 �������"  "�  "�����"
	 ������"5*1sequential/densenet169/conv2_block2_2_conv/Conv2D"
 �������"
*output" "*[200,32,8,8]"�  "�����"
	 ��؄��"5*1sequential/densenet169/conv2_block2_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  "+R������"��" *$$1""8v"�  "���ղ"
	 ������"5*1sequential/densenet169/conv2_block2_2_conv/Conv2D"
 �������"*temp" "*
[19759104]"�  "+[�������"�" *$$1""8w"�  "+\�������"�" *$$1""8x"�  "+]�������"�" *$$1""8y"�  "+^������"��" *$$1""8z"�  "�е��"
	 ������"5*1sequential/densenet169/conv2_block2_2_conv/Conv2D"
 �������"  "�  "����"
	 ��؄��"5*1sequential/densenet169/conv2_block2_2_conv/Conv2D"
 �������"  "�  "�����"
	 ������"5*1sequential/densenet169/conv2_block2_concat/concat"
 �������"
*output" "*
	 ������"
  "  "�  "�����"
	 ������"=*9sequential/densenet169/conv2_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv2_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����"
	 �����"=*9sequential/densenet169/conv2_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����"
	 �����"=*9sequential/densenet169/conv2_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��З�"
	 �����"=*9sequential/densenet169/conv2_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�������"��" *$$1"!*size:512 dest:0 async:1"�  "BU�������"��" *$$1"!*size:512 dest:0 async:1"�  "+V�������"��" *$$1""8}"�  "+W�Ǚ����"��" *$$1""8~"�  "�Ȝ��"
	 �؀���";*7sequential/densenet169/conv2_block3_0_bn/AssignNewValue"
 �������"  "�  "��ڃ�"
	 �܀���"=*9sequential/densenet169/conv2_block3_0_bn/AssignNewValue_1"
 �������"  "�  "�Ȏ��"
	 ������"5*1sequential/densenet169/conv2_block3_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv2_block3_1_conv/Conv2D"
 �������"*temp" "*
	 �����"5*1sequential/densenet169/conv2_block3_1_conv/Conv2D"
 �������"*temp" "	*[392]"�  ",S�����מ"��" *$$1""8�"�  ",Z���Ё�"��" *$$1""8�"�  "���Δ"
	 �����"5*1sequential/densenet169/conv2_block3_1_conv/Conv2D"
 �������"  "�  "�����"
	 ������"5*1sequential/densenet169/conv2_block3_1_conv/Conv2D"
 �������"  "�  "����"
(��?" ����" ���" ���"
	 ������"=*9sequential/densenet169/conv2_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "*
(��?" ����" �" �"
	 �����"=*9sequential/densenet169/conv2_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���"
(��?" ����" �" �"
	 �����"=*9sequential/densenet169/conv2_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����"
	 �����"=*9sequential/densenet169/conv2_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����"
	 �����"=*9sequential/densenet169/conv2_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�����"��" *$$1"!*size:512 dest:0 async:1"�  "BU�������"��" *$$1"!*size:512 dest:0 async:1"�  ",V�������"��" *$$1""8�"�  ",W�������"��" *$$1""8�"�  "����"
	 �耥��";*7sequential/densenet169/conv2_block3_1_bn/AssignNewValue"
 �������"  "�  "�����"
(��?" ����" �" �"
	 �쀥��"=*9sequential/densenet169/conv2_block3_1_bn/AssignNewValue_1"
 �������"  "�  "�����"
	 ������"5*1sequential/densenet169/conv2_block3_2_conv/Conv2D"
 �������"
*output" "*[200,32,8,8]"�  "��ش"
	 ��؄��"5*1sequential/densenet169/conv2_block3_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R��ܵ���"��" *$$1""8�"�  "��ݨ�"
	 �����"5*1sequential/densenet169/conv2_block3_2_conv/Conv2D"
 �������"*temp" "*
[19759104]"�  ",[��ݽ�ј"��" *$$1""8�"�  ",\�������"��" *$$1""8�"�  ",]�������"��" *$$1""8�"�  ",^������"��" *$$1""8�"�  "�����"
	 �����"5*1sequential/densenet169/conv2_block3_2_conv/Conv2D"
 �������"  "�  "�����"
	 ��؄��"5*1sequential/densenet169/conv2_block3_2_conv/Conv2D"
 �������"  "�  "��К�"
	 ������"5*1sequential/densenet169/conv2_block3_concat/concat"
 �������"
*output" "*
	 ������"
  "  "�  "��ߨ�"
	 ������"=*9sequential/densenet169/conv2_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv2_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[160]"�  "��Ǝ�"
	 �����"=*9sequential/densenet169/conv2_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[160]"�  "�����"
	 ������"=*9sequential/densenet169/conv2_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[160]"�  "�и��"
	 ������"=*9sequential/densenet169/conv2_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[160]"�  "BU������"��" *$$1"!*size:640 dest:0 async:1"�  "BU����Ȣ�"��" *$$1"!*size:640 dest:0 async:1"�  ",V�������"��" *$$1""8�"�  ",W�����͜"��" *$$1""8�"�  "�����"
	 ������";*7sequential/densenet169/conv2_block4_0_bn/AssignNewValue"
 �������"  "�  "�����"
	 ������"=*9sequential/densenet169/conv2_block4_0_bn/AssignNewValue_1"
 �������"  "�  "�����"
	 �����"5*1sequential/densenet169/conv2_block4_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv2_block4_1_conv/Conv2D"
 �������"*temp" "*
	 �����"5*1sequential/densenet169/conv2_block4_1_conv/Conv2D"
 �������"*temp" "	*[392]"�  ",S�������"��" *$$1""8�"�  ",Z������"��" *$$1""8�"�  "��ǿ�"
	 �����"5*1sequential/densenet169/conv2_block4_1_conv/Conv2D"
 �������"  "�  "�����"
	 ������"5*1sequential/densenet169/conv2_block4_1_conv/Conv2D"
 �������"  "�  "���Í"
	 �ҁ���"=*9sequential/densenet169/conv2_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv2_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�؉Ŏ"
	 ������"=*9sequential/densenet169/conv2_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����"
	 ������"=*9sequential/densenet169/conv2_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�躢�"
	 �Ī���"=*9sequential/densenet169/conv2_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�᧐���"��" *$$1"!*size:512 dest:0 async:1"�  "BUȎ���ב"��" *$$1"!*size:512 dest:0 async:1"�  ",V�����"��" *$$1""8�"�  ",W����В�"��" *$$1""8�"�  "���ݡ"
	 ������";*7sequential/densenet169/conv2_block4_1_bn/AssignNewValue"
 �������"  "�  "�����"
	 ������"=*9sequential/densenet169/conv2_block4_1_bn/AssignNewValue_1"
 �������"  "�  "�����"
	 ������"5*1sequential/densenet169/conv2_block4_2_conv/Conv2D"
 �������"
*output" "*[200,32,8,8]"�  "�蕓�"
	 ��؄��"5*1sequential/densenet169/conv2_block4_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����п�"��" *$$1""8�"�  "�ࠗ�"
	 �ґ���"5*1sequential/densenet169/conv2_block4_2_conv/Conv2D"
 �������"*temp" "*
[19759104]"�  ",[��¯�֙"��" *$$1""8�"�  ",\؟���Ϯ"��" *$$1""8�"�  ",]��ʴ���"��" *$$1""8�"�  ",^Н����"��" *$$1""8�"�  "�����"
	 �ґ���"5*1sequential/densenet169/conv2_block4_2_conv/Conv2D"
 �������"  "�  "���"
	 ��؄��"5*1sequential/densenet169/conv2_block4_2_conv/Conv2D"
 �������"  "�  "����"
�Rς�?" ����" ���" ���"
	 �ґ���"5*1sequential/densenet169/conv2_block4_concat/concat"
 �������"
*output" "*
	 ������"
  "  "�  "�����"
	 �����"=*9sequential/densenet169/conv2_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �Ы���"=*9sequential/densenet169/conv2_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[192]"�  "����"
	 �⬑��"=*9sequential/densenet169/conv2_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[192]"�  "�����"
	 ������"=*9sequential/densenet169/conv2_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[192]"�  "�����"
	 ���"=*9sequential/densenet169/conv2_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[192]"�  "BU����ȭ�"��" *$$1"!*size:768 dest:0 async:1"�  "BU�Ӊ����"��" *$$1"!*size:768 dest:0 async:1"�  ",V����ۼ"��" *$$1""8�"�  ",W�����ł"��" *$$1""8�"�  "����"
	 ������";*7sequential/densenet169/conv2_block5_0_bn/AssignNewValue"
 �������"  "�  "�����"
	 ������"=*9sequential/densenet169/conv2_block5_0_bn/AssignNewValue_1"
 �������"  "�  "�����"
	 ������"5*1sequential/densenet169/conv2_block5_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv2_block5_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv2_block5_1_conv/Conv2D"
 �������"*temp" "	*[392]"�  ",S�����ܱ"��" *$$1""8�"�  ",T�������"��" *$$1""8�"�  "�����"
	 ������"5*1sequential/densenet169/conv2_block5_1_conv/Conv2D"
 �������"  "�  "����"
	 ������"5*1sequential/densenet169/conv2_block5_1_conv/Conv2D"
 �������"  "�  "�����"
	 ��ѿ��"=*9sequential/densenet169/conv2_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv2_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����"
	 ������"=*9sequential/densenet169/conv2_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��¢�"
	 ������"=*9sequential/densenet169/conv2_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����"
	 ������"=*9sequential/densenet169/conv2_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�������"��" *$$1"!*size:512 dest:0 async:1"�  "BU����؛�"��" *$$1"!*size:512 dest:0 async:1"�  ",V���� ���"��" *$$1""8�"�  ",W�ֆ� ��"��" *$$1""8�"�  "���؊ "
	 ������";*7sequential/densenet169/conv2_block5_1_bn/AssignNewValue"
 �������"  "�  "�Ȭƌ "
	 ������"=*9sequential/densenet169/conv2_block5_1_bn/AssignNewValue_1"
 �������"  "�  "���Ǝ "
	 ������"5*1sequential/densenet169/conv2_block5_2_conv/Conv2D"
 �������"
*output" "*[200,32,8,8]"�  "���Տ "
	 ��؄��"5*1sequential/densenet169/conv2_block5_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R��Ԑ �ͩ"��" *$$1""8�"�  "��陖 "
	 ������"5*1sequential/densenet169/conv2_block5_2_conv/Conv2D"
 �������"*temp" "*
[19759104]"�  ",[�н� ���"��" *$$1""8�"�  ",\��ٛ ��"��" *$$1""8�"�  ",]�Ë� ���"��" *$$1""8�"�  ",^���� ���"��" *$$1""8�"�  "���Т "
	 ������"5*1sequential/densenet169/conv2_block5_2_conv/Conv2D"
 �������"  "�  "��֢� "
	 ��؄��"5*1sequential/densenet169/conv2_block5_2_conv/Conv2D"
 �������"  "�  "����� "
	 ������"5*1sequential/densenet169/conv2_block5_concat/concat"
 �������"
*output" "*
	 ������"
  "  "�  "��ؚ� "
	 ������"=*9sequential/densenet169/conv2_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �Ā���"=*9sequential/densenet169/conv2_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[224]"�  "�к�� "
	 �؀���"=*9sequential/densenet169/conv2_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[224]"�  "���� "
	 �耥��"=*9sequential/densenet169/conv2_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[224]"�  "���� "
	 ������"=*9sequential/densenet169/conv2_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[224]"�  "BU��� ���"��" *$$1"!*size:896 dest:0 async:1"�  "BU��� �Ě"��" *$$1"!*size:896 dest:0 async:1"�  ",V���� ���"��" *$$1""8�"�  ",W�ԟ� ���"��" *$$1""8�"�  "��ԋ� "
	 �ԁ���";*7sequential/densenet169/conv2_block6_0_bn/AssignNewValue"
 �������"  "�  "����� "
	 ������"=*9sequential/densenet169/conv2_block6_0_bn/AssignNewValue_1"
 �������"  "�  "����� "
	 ������"5*1sequential/densenet169/conv2_block6_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv2_block6_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv2_block6_1_conv/Conv2D"
 �������"*temp" "	*[392]"�  ",S��� �ߖ"��" *$$1""8�"�  ",T���� ���"��" *$$1""8�"�  "�Ф�� "
	 ������"5*1sequential/densenet169/conv2_block6_1_conv/Conv2D"
 �������"  "�  "����� "
	 ������"5*1sequential/densenet169/conv2_block6_1_conv/Conv2D"
 �������"  "�  "����� "
	 ������"=*9sequential/densenet169/conv2_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv2_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����� "
	 ������"=*9sequential/densenet169/conv2_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����� "
	 ������"=*9sequential/densenet169/conv2_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�Ȫ�� "
	 ������"=*9sequential/densenet169/conv2_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BUȞ�� ���"��" *$$1"!*size:512 dest:0 async:1"�  "BU�ן� �"��" *$$1"!*size:512 dest:0 async:1"�  ",V���� ���"��" *$$1""8�"�  ",W���� �"��" *$$1""8�"�  "����� "
	 �聥��";*7sequential/densenet169/conv2_block6_1_bn/AssignNewValue"
 �������"  "�  "�Ȃ�� "
	 �쁥��"=*9sequential/densenet169/conv2_block6_1_bn/AssignNewValue_1"
 �������"  "�  "����� "
	 ������"5*1sequential/densenet169/conv2_block6_2_conv/Conv2D"
 �������"
*output" "*[200,32,8,8]"�  "��̦� "
	 ��؄��"5*1sequential/densenet169/conv2_block6_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R���� ���"��" *$$1""8�"�  "����� "
	 ������"5*1sequential/densenet169/conv2_block6_2_conv/Conv2D"
 �������"*temp" "*
[19759104]"�  ",[���� �և"��" *$$1""8�"�  ",\�·�!���"��" *$$1""8�"�  ",]��܄!���"��" *$$1""8�"�  ",^����!���"��" *$$1""8�"�  "����!"
	 ������"5*1sequential/densenet169/conv2_block6_2_conv/Conv2D"
 �������"  "�  "��´�!"
	 ��؄��"5*1sequential/densenet169/conv2_block6_2_conv/Conv2D"
 �������"  "�  "��ϗ�!"
	 ������"5*1sequential/densenet169/conv2_block6_concat/concat"
 �������"
*output" "*
	 ������"
  "  "�  "�����!"
	 ������"4*0sequential/densenet169/pool2_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"4*0sequential/densenet169/pool2_bn/FusedBatchNormV3"
 �������"
*output" "	*[256]"�  "�����!"
	 �ԁ���"4*0sequential/densenet169/pool2_bn/FusedBatchNormV3"
 �������"
*output" "	*[256]"�  "�����!"
	 ������"4*0sequential/densenet169/pool2_bn/FusedBatchNormV3"
 �������"
*output" "	*[256]"�  "���ݗ!"
	 ����"4*0sequential/densenet169/pool2_bn/FusedBatchNormV3"
 �������"
*output" "	*[256]"�  "CU���!���"��" *$$1"!*size:1024 dest:0 async:1"�  "CU��՜!�ڐ"��" *$$1"!*size:1024 dest:0 async:1"�  ",Vȷ��!���"��" *$$1""8�"�  ",W����!�݂"��" *$$1""8�"�  "�����!"
	 ������"2*.sequential/densenet169/pool2_bn/AssignNewValue"
 �������"  "�  "����!"
	 ������"4*0sequential/densenet169/pool2_bn/AssignNewValue_1"
 �������"  "�  "����!"
	 ������",*(sequential/densenet169/pool2_conv/Conv2D"
 �������"
*output" "*
	 ������",*(sequential/densenet169/pool2_conv/Conv2D"
 �������"*temp" "*
	 ������",*(sequential/densenet169/pool2_conv/Conv2D"
 �������"*temp" "	*[392]"�  ",S����!أ�"��" *$$1""8�"�  ",Tث�!Ъ�"��" *$$1""8�"�  "����!"
	 ������",*(sequential/densenet169/pool2_conv/Conv2D"
 �������"  "�  "����!"
	 ������",*(sequential/densenet169/pool2_conv/Conv2D"
 �������"  "�  "�ؽ��!"
	 ������"-*)sequential/densenet169/pool2_pool/AvgPool"
 �������"
*output" "*
	 ������"
  "  "�  "�����!"
	 ��؄��"=*9sequential/densenet169/conv3_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv3_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����!"
	 ������"=*9sequential/densenet169/conv3_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����!"
	 ����"=*9sequential/densenet169/conv3_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����!"
	 ������"=*9sequential/densenet169/conv3_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�Қ�!���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����!���"��" *$$1"!*size:512 dest:0 async:1"�  ",`�ڦ�!���"��" *$$1""8�"�  ",W����!���"��" *$$1""8�"�  "��ٟ�!"
	 ������";*7sequential/densenet169/conv3_block1_0_bn/AssignNewValue"
 �������"  "�  "��ø�!"
	 ������"=*9sequential/densenet169/conv3_block1_0_bn/AssignNewValue_1"
 �������"  "�  "�����!"
	 �����"5*1sequential/densenet169/conv3_block1_1_conv/Conv2D"
 �������"
*output" "*
	 ��ɫ��"5*1sequential/densenet169/conv3_block1_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv3_block1_1_conv/Conv2D"
 �������"*temp" "	*[104]"�  ",S����!��"��" *$$1""8�"�  ",Z����!���"��" *$$1""8�"�  "�����!"
	 ������"5*1sequential/densenet169/conv3_block1_1_conv/Conv2D"
 �������"  "�  "�����!"
	 ��ɫ��"5*1sequential/densenet169/conv3_block1_1_conv/Conv2D"
 �������"  "�  "�����!"
	 ��ɫ��"=*9sequential/densenet169/conv3_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv3_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���Ѐ""
	 ������"=*9sequential/densenet169/conv3_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����""
	 ������"=*9sequential/densenet169/conv3_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����""
	 ������"=*9sequential/densenet169/conv3_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����"���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����"���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����"���"��" *$$1""8�"�  ",W����"���"��" *$$1""8�"�  "�����""
	 �Ƃ���";*7sequential/densenet169/conv3_block1_1_bn/AssignNewValue"
 �������"  "�  "���Ε""
	 �ʂ���"=*9sequential/densenet169/conv3_block1_1_bn/AssignNewValue_1"
 �������"  "�  "���՗""
	 ��̟��"5*1sequential/densenet169/conv3_block1_2_conv/Conv2D"
 �������"
*output" "*[200,32,4,4]"�  "��ޘ""
	 �����"5*1sequential/densenet169/conv3_block1_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",RȰߙ"���"��" *$$1""8�"�  "�����""
	 ������"5*1sequential/densenet169/conv3_block1_2_conv/Conv2D"
 �������"*temp" "
	 ������"5*1sequential/densenet169/conv3_block1_2_conv/Conv2D"
 �������"  "�  "�𷾫""
	 �����"5*1sequential/densenet169/conv3_block1_2_conv/Conv2D"
 �������"  "�  "��¢�""
	 �����"5*1sequential/densenet169/conv3_block1_concat/concat"
 �������"
*output" "*
	 ��̟��"
  "  "�  "����""
	 ��܅��"=*9sequential/densenet169/conv3_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �Ƃ���"=*9sequential/densenet169/conv3_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[160]"�  "���ܸ""
	 ������"=*9sequential/densenet169/conv3_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[160]"�  "�����""
	 ������"=*9sequential/densenet169/conv3_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[160]"�  "�����""
	 �ƈ���"=*9sequential/densenet169/conv3_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[160]"�  "BU�н�"��"��" *$$1"!*size:640 dest:0 async:1"�  "BU�ꣾ"���"��" *$$1"!*size:640 dest:0 async:1"�  ",`����"���"��" *$$1""8�"�  ",Wذ��"���"��" *$$1""8�"�  "��Ͼ�""
	 �ւ���";*7sequential/densenet169/conv3_block2_0_bn/AssignNewValue"
 �������"  "�  "�����""
	 ������"=*9sequential/densenet169/conv3_block2_0_bn/AssignNewValue_1"
 �������"  "�  "�����""
	 �����"5*1sequential/densenet169/conv3_block2_1_conv/Conv2D"
 �������"
*output" "*
	 ��̟��"5*1sequential/densenet169/conv3_block2_1_conv/Conv2D"
 �������"*temp" "*
	 �ւ���"5*1sequential/densenet169/conv3_block2_1_conv/Conv2D"
 �������"*temp" "	*[104]"�  ",S���"�ב"��" *$$1""8�"�  ",Z����"���"��" *$$1""8�"�  "�����""
	 �ւ���"5*1sequential/densenet169/conv3_block2_1_conv/Conv2D"
 �������"  "�  "�����""
	 ��̟��"5*1sequential/densenet169/conv3_block2_1_conv/Conv2D"
 �������"  "�  "�����""
	 ��Ւ��"=*9sequential/densenet169/conv3_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �ւ���"=*9sequential/densenet169/conv3_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����""
	 �ڂ���"=*9sequential/densenet169/conv3_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����""
	 ������"=*9sequential/densenet169/conv3_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����""
	 ������"=*9sequential/densenet169/conv3_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����"���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����"���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����"���"��" *$$1""8�"�  ",W����"���"��" *$$1""8�"�  "�����#"
	 �肥��";*7sequential/densenet169/conv3_block2_1_bn/AssignNewValue"
 �������"  "�  "�Ȩ��#"
	 �삥��"=*9sequential/densenet169/conv3_block2_1_bn/AssignNewValue_1"
 �������"  "�  "�����#"
	 ��̟��"5*1sequential/densenet169/conv3_block2_2_conv/Conv2D"
 �������"
*output" "*[200,32,4,4]"�  "��ᣇ#"
	 ������"5*1sequential/densenet169/conv3_block2_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R���#���"��" *$$1""8�"�  "���͔#"
	 ������"5*1sequential/densenet169/conv3_block2_2_conv/Conv2D"
 �������"*temp" "
	 ������"5*1sequential/densenet169/conv3_block2_2_conv/Conv2D"
 �������"  "�  "��ţ#"
	 ������"5*1sequential/densenet169/conv3_block2_2_conv/Conv2D"
 �������"  "�  "���#"
	 ������"5*1sequential/densenet169/conv3_block2_concat/concat"
 �������"
*output" "*
	 ��̟��"
  "  "�  "�����#"
	 ������"=*9sequential/densenet169/conv3_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �肥��"=*9sequential/densenet169/conv3_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[192]"�  "�����#"
	 ������"=*9sequential/densenet169/conv3_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[192]"�  "�����#"
	 ������"=*9sequential/densenet169/conv3_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[192]"�  "��ޱ#"
	 �⋥��"=*9sequential/densenet169/conv3_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[192]"�  "BU����#��"��" *$$1"!*size:768 dest:0 async:1"�  "BU�ꈷ#���"��" *$$1"!*size:768 dest:0 async:1"�  ",`��Ҽ#���"��" *$$1""8�"�  ",W����#���"��" *$$1""8�"�  "�����#"
	 ������";*7sequential/densenet169/conv3_block3_0_bn/AssignNewValue"
 �������"  "�  "�ػ��#"
	 ������"=*9sequential/densenet169/conv3_block3_0_bn/AssignNewValue_1"
 �������"  "�  "�����#"
	 ����"5*1sequential/densenet169/conv3_block3_1_conv/Conv2D"
 �������"
*output" "*
	 ��̟��"5*1sequential/densenet169/conv3_block3_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv3_block3_1_conv/Conv2D"
 �������"*temp" "	*[104]"�  ",SО��#���"��" *$$1""8�"�  ",T����#���"��" *$$1""8�"�  "�����#"
	 ������"5*1sequential/densenet169/conv3_block3_1_conv/Conv2D"
 �������"  "�  "�����#"
	 ��̟��"5*1sequential/densenet169/conv3_block3_1_conv/Conv2D"
 �������"  "�  "�����#"
	 ����"=*9sequential/densenet169/conv3_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv3_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����#"
	 ������"=*9sequential/densenet169/conv3_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�Ђ��#"
	 �ƌ���"=*9sequential/densenet169/conv3_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����#"
	 �ʌ���"=*9sequential/densenet169/conv3_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����#���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����#���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����#І�"��" *$$1""8�"�  ",W����#���"��" *$$1""8�"�  "����#"
	 ������";*7sequential/densenet169/conv3_block3_1_bn/AssignNewValue"
 �������"  "�  "�����#"
	 ������"=*9sequential/densenet169/conv3_block3_1_bn/AssignNewValue_1"
 �������"  "�  "�����$"
	 ��̟��"5*1sequential/densenet169/conv3_block3_2_conv/Conv2D"
 �������"
*output" "*[200,32,4,4]"�  "���݂$"
	 ������"5*1sequential/densenet169/conv3_block3_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����$�̱"��" *$$1""8�"�  "�����$"
	 ������"5*1sequential/densenet169/conv3_block3_2_conv/Conv2D"
 �������"*temp" "
	 ������"5*1sequential/densenet169/conv3_block3_2_conv/Conv2D"
 �������"  "�  "��ޗ�$"
	 ������"5*1sequential/densenet169/conv3_block3_2_conv/Conv2D"
 �������"  "�  "�����$"
	 ������"5*1sequential/densenet169/conv3_block3_concat/concat"
 �������"
*output" "*
��?" ����" ��" ��"
	 ��̟��"
  "  "�  "����$"
	 ������"=*9sequential/densenet169/conv3_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv3_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[224]"�  "�Д�$"
	 ������"=*9sequential/densenet169/conv3_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[224]"�  "�����$"
	 ������"=*9sequential/densenet169/conv3_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[224]"�  "���ƥ$"
	 ������"=*9sequential/densenet169/conv3_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[224]"�  "BU��Ӧ$��"��" *$$1"!*size:896 dest:0 async:1"�  "BU��٪$��"��" *$$1"!*size:896 dest:0 async:1"�  ",`��ͯ$���"��" *$$1""8�"�  ",W��ݶ$���"��" *$$1""8�"�  "�����$"
	 ������";*7sequential/densenet169/conv3_block4_0_bn/AssignNewValue"
 �������"  "�  "��ݯ�$"
	 ������"=*9sequential/densenet169/conv3_block4_0_bn/AssignNewValue_1"
 �������"  "�  "�����$"
	 ������"5*1sequential/densenet169/conv3_block4_1_conv/Conv2D"
 �������"
*output" "*
	 ��̟��"5*1sequential/densenet169/conv3_block4_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv3_block4_1_conv/Conv2D"
 �������"*temp" "	*[104]"�  ",S����$���"��" *$$1""8�"�  ",T����$���"��" *$$1""8�"�  "�����$"
	 ������"5*1sequential/densenet169/conv3_block4_1_conv/Conv2D"
 �������"  "�  "�����$"
	 ��̟��"5*1sequential/densenet169/conv3_block4_1_conv/Conv2D"
 �������"  "�  "�����$"
	 ������"=*9sequential/densenet169/conv3_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv3_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����$"
	 ������"=*9sequential/densenet169/conv3_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����$"
	 �܏���"=*9sequential/densenet169/conv3_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����$"
	 ������"=*9sequential/densenet169/conv3_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����$،�"��" *$$1"!*size:512 dest:0 async:1"�  "BU����$ȇ�"��" *$$1"!*size:512 dest:0 async:1"�  ",`����$�Ȇ"��" *$$1""8�"�  ",WЀ��$��"��" *$$1""8�"�  "��ŏ�%"
	 ������";*7sequential/densenet169/conv3_block4_1_bn/AssignNewValue"
 �������"  "�  "�����%"
	 ������"=*9sequential/densenet169/conv3_block4_1_bn/AssignNewValue_1"
 �������"  "�  "��Ȧ�%"
	 ��̟��"5*1sequential/densenet169/conv3_block4_2_conv/Conv2D"
 �������"
*output" "*[200,32,4,4]"�  "��삊%"
	 ������"5*1sequential/densenet169/conv3_block4_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R�翋%���"��" *$$1""8�"�  "�����%"
	 ������"5*1sequential/densenet169/conv3_block4_2_conv/Conv2D"
 �������"*temp" "
	 ������"5*1sequential/densenet169/conv3_block4_2_conv/Conv2D"
 �������"  "�  "��㈣%"
	 ������"5*1sequential/densenet169/conv3_block4_2_conv/Conv2D"
 �������"  "�  "��ȯ�%"
	 ������"5*1sequential/densenet169/conv3_block4_concat/concat"
 �������"
*output" "*
	 ��̟��"
  "  "�  "���ޯ%"
	 ������"=*9sequential/densenet169/conv3_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv3_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[256]"�  "�����%"
	 ������"=*9sequential/densenet169/conv3_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[256]"�  "���ڱ%"
	 �Ē���"=*9sequential/densenet169/conv3_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[256]"�  "��曲%"
	 ������"=*9sequential/densenet169/conv3_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[256]"�  "CU��Ŵ%���"��" *$$1"!*size:1024 dest:0 async:1"�  "CU�携%��"��" *$$1"!*size:1024 dest:0 async:1"�  ",`����%���"��" *$$1""8�"�  ",W����%���"��" *$$1""8�"�  "�Ȏ��%"
	 �҃���";*7sequential/densenet169/conv3_block5_0_bn/AssignNewValue"
 �������"  "�  "�����%"
	 ������"=*9sequential/densenet169/conv3_block5_0_bn/AssignNewValue_1"
 �������"  "�  "�蜡�%"
	 ������"5*1sequential/densenet169/conv3_block5_1_conv/Conv2D"
 �������"
*output" "*
	 ��̟��"5*1sequential/densenet169/conv3_block5_1_conv/Conv2D"
 �������"*temp" "*
	 �҃���"5*1sequential/densenet169/conv3_block5_1_conv/Conv2D"
 �������"*temp" "	*[104]"�  ",S�Ĳ�%���"��" *$$1""8�"�  ",T����%���"��" *$$1""8�"�  "����%"
	 �҃���"5*1sequential/densenet169/conv3_block5_1_conv/Conv2D"
 �������"  "�  "�����%"
	 ��̟��"5*1sequential/densenet169/conv3_block5_1_conv/Conv2D"
 �������"  "�  "�����%"
	 ������"=*9sequential/densenet169/conv3_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �҃���"=*9sequential/densenet169/conv3_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����%"
	 �փ���"=*9sequential/densenet169/conv3_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����%"
	 �Γ���"=*9sequential/densenet169/conv3_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ܘ�%"
	 �ғ���"=*9sequential/densenet169/conv3_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�֬�%���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����%���"��" *$$1"!*size:512 dest:0 async:1"�  ",`ح��%��"��" *$$1""8�"�  ",W���&���"��" *$$1""8�"�  "���Å&"
	 �ꃥ��";*7sequential/densenet169/conv3_block5_1_bn/AssignNewValue"
 �������"  "�  "�����&"
	 ���"=*9sequential/densenet169/conv3_block5_1_bn/AssignNewValue_1"
 �������"  "�  "���&"
	 ��̟��"5*1sequential/densenet169/conv3_block5_2_conv/Conv2D"
 �������"
*output" "*[200,32,4,4]"�  "�����&"
	 ������"5*1sequential/densenet169/conv3_block5_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����&�Ǧ"��" *$$1""8�"�  "���ې&"
	 ������"5*1sequential/densenet169/conv3_block5_2_conv/Conv2D"
 �������"*temp" "
	 ������"5*1sequential/densenet169/conv3_block5_2_conv/Conv2D"
 �������"  "�  "�ظ�&"
	 ������"5*1sequential/densenet169/conv3_block5_2_conv/Conv2D"
 �������"  "�  "��˚�&"
	 ������"5*1sequential/densenet169/conv3_block5_concat/concat"
 �������"
*output" "*
	 ��̟��"
  "  "�  "��ꌪ&"
	 ������"=*9sequential/densenet169/conv3_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "*
"
	 �����"=*9sequential/densenet169/conv3_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[288]"�  "�����&"
	 �聥��"=*9sequential/densenet169/conv3_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[288]"�  "���&"
	 �ĕ���"=*9sequential/densenet169/conv3_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[288]"�  "���ˬ&"
	 ������"=*9sequential/densenet169/conv3_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[288]"�  "CU��ޭ&���"��" *$$1"!*size:1152 dest:0 async:1"�  "CU���&�Ē"��" *$$1"!*size:1152 dest:0 async:1"�  ",`��ն&��"��" *$$1""8�"�  ",W��ռ&���"��" *$$1""8�"�  "�����&"
"
	 ������";*7sequential/densenet169/conv3_block6_0_bn/AssignNewValue"
 �������"  "�  "�����&"
"
	 ������"=*9sequential/densenet169/conv3_block6_0_bn/AssignNewValue_1"
 �������"  "�  "�И��&"
	 ������"5*1sequential/densenet169/conv3_block6_1_conv/Conv2D"
 �������"
*output" "*
	 ��̟��"5*1sequential/densenet169/conv3_block6_1_conv/Conv2D"
 �������"*temp" "*
	 �ꃥ��"5*1sequential/densenet169/conv3_block6_1_conv/Conv2D"
 �������"*temp" "	*[104]"�  ",S����&���"��" *$$1""8�"�  ",T����&���"��" *$$1""8�"�  "�����&"
	 �ꃥ��"5*1sequential/densenet169/conv3_block6_1_conv/Conv2D"
 �������"  "�  "�����&"
	 ��̟��"5*1sequential/densenet169/conv3_block6_1_conv/Conv2D"
 �������"  "�  "�����&"
	 ������"=*9sequential/densenet169/conv3_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �ꃥ��"=*9sequential/densenet169/conv3_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����&"
	 ���"=*9sequential/densenet169/conv3_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����&"
	 ������"=*9sequential/densenet169/conv3_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ŧ�&"
	 ������"=*9sequential/densenet169/conv3_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����&���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����&�ڃ"��" *$$1"!*size:512 dest:0 async:1"�  ",`����&���"��" *$$1""8�"�  ",W����&���"��" *$$1""8�"�  "�����&"
	 ������";*7sequential/densenet169/conv3_block6_1_bn/AssignNewValue"
 �������"  "�  "�����&"
	 ������"=*9sequential/densenet169/conv3_block6_1_bn/AssignNewValue_1"
 �������"  "�  "��ְ�&"
	 ��̟��"5*1sequential/densenet169/conv3_block6_2_conv/Conv2D"
 �������"
*output" "*[200,32,4,4]"�  "�����&"
	 ������"5*1sequential/densenet169/conv3_block6_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����&���"��" *$$1""8�"�  "�����&"
	 ������"5*1sequential/densenet169/conv3_block6_2_conv/Conv2D"
 �������"*temp" "
	 ������"5*1sequential/densenet169/conv3_block6_2_conv/Conv2D"
 �������"  "�  "�У��'"
	 ������"5*1sequential/densenet169/conv3_block6_2_conv/Conv2D"
 �������"  "�  "�����'"
	 ������"5*1sequential/densenet169/conv3_block6_concat/concat"
 �������"
*output" "*
	 ��̟��"
  "  "�  "�����'"
	 ������"=*9sequential/densenet169/conv3_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "*
" �
"
	 ������"=*9sequential/densenet169/conv3_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[320]"�  "�����'"
" �
"
	 ������"=*9sequential/densenet169/conv3_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[320]"�  "�𬳓'"
" �
"
	 ������"=*9sequential/densenet169/conv3_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[320]"�  "���ޓ'"
" �"
	 ������"=*9sequential/densenet169/conv3_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[320]"�  "CU���'���"��" *$$1"!*size:1280 dest:0 async:1"�  "CU���'���"��" *$$1"!*size:1280 dest:0 async:1"�  ",`���'��"��" *$$1""8�"�  ",W��ܣ'���"��" *$$1""8�"�  "�����'"
" �"
	 ������";*7sequential/densenet169/conv3_block7_0_bn/AssignNewValue"
 �������"  "�  "��ɨ'"
" �
"
	 ������"=*9sequential/densenet169/conv3_block7_0_bn/AssignNewValue_1"
 �������"  "�  "���ī'"
	 ������"5*1sequential/densenet169/conv3_block7_1_conv/Conv2D"
 �������"
*output" "*
" ��
"
	 ��̟��"5*1sequential/densenet169/conv3_block7_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv3_block7_1_conv/Conv2D"
 �������"*temp" "	*[104]"�  ",Sأ��'ȝ�"��" *$$1""8�"�  ",T�ƺ'���"��" *$$1""8�"�  "�����'"
	 ������"5*1sequential/densenet169/conv3_block7_1_conv/Conv2D"
 �������"  "�  "��̼�'"
" ��
"
	 ��̟��"5*1sequential/densenet169/conv3_block7_1_conv/Conv2D"
 �������"  "�  "�����'"
	 ������"=*9sequential/densenet169/conv3_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv3_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����'"
	 ������"=*9sequential/densenet169/conv3_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����'"
	 �蔥��"=*9sequential/densenet169/conv3_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��۫�'"
	 �씥��"=*9sequential/densenet169/conv3_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU���'ȹ�"��" *$$1"!*size:512 dest:0 async:1"�  "BUЕ��'���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����'��"��" *$$1""8�"�  ",W�ɣ�'ؿ�"��" *$$1""8�"�  "�����'"
	 ����";*7sequential/densenet169/conv3_block7_1_bn/AssignNewValue"
 �������"  "�  "�����'"
	 �Ƅ���"=*9sequential/densenet169/conv3_block7_1_bn/AssignNewValue_1"
 �������"  "�  "�����'"
	 ��̟��"5*1sequential/densenet169/conv3_block7_2_conv/Conv2D"
 �������"
*output" "*[200,32,4,4]"�  "�Ю��'"
	 ������"5*1sequential/densenet169/conv3_block7_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",RȄ��'���"��" *$$1""8�"�  "�����'"
	 ������"5*1sequential/densenet169/conv3_block7_2_conv/Conv2D"
 �������"*temp" "
	 ������"5*1sequential/densenet169/conv3_block7_2_conv/Conv2D"
 �������"  "�  "�����'"
	 ������"5*1sequential/densenet169/conv3_block7_2_conv/Conv2D"
 �������"  "�  "�����'"
	 ������"5*1sequential/densenet169/conv3_block7_concat/concat"
 �������"
*output" "*
	 ��̟��"
  "  "�  "�����'"
	 ������"=*9sequential/densenet169/conv3_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv3_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[352]"�  "�����'"
	 ������"=*9sequential/densenet169/conv3_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[352]"�  "�д��'"
	 ������"=*9sequential/densenet169/conv3_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[352]"�  "�����'"
	 �����"=*9sequential/densenet169/conv3_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[352]"�  "CU�ɺ�'���"��" *$$1"!*size:1408 dest:0 async:1"�  "CU���'��"��" *$$1"!*size:1408 dest:0 async:1"�  ",`��Ѓ(�ׄ"��" *$$1""8�"�  ",W��ˉ(���"��" *$$1""8�"�  "����("
	 �؄���";*7sequential/densenet169/conv3_block8_0_bn/AssignNewValue"
 �������"  "�  "�����("
	 ������"=*9sequential/densenet169/conv3_block8_0_bn/AssignNewValue_1"
 �������"  "�  "�����("
	 ������"5*1sequential/densenet169/conv3_block8_1_conv/Conv2D"
 �������"
*output" "*
	 ��̟��"5*1sequential/densenet169/conv3_block8_1_conv/Conv2D"
 �������"*temp" "*
	 ����"5*1sequential/densenet169/conv3_block8_1_conv/Conv2D"
 �������"*temp" "	*[104]"�  ",S��͝(���"��" *$$1""8�"�  ",T����(���"��" *$$1""8�"�  "���ң("
	 ����"5*1sequential/densenet169/conv3_block8_1_conv/Conv2D"
 �������"  "�  "�����("
	 ��̟��"5*1sequential/densenet169/conv3_block8_1_conv/Conv2D"
 �������"  "�  "�����("
	 ������"=*9sequential/densenet169/conv3_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ����"=*9sequential/densenet169/conv3_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���̧("
	 �Ƅ���"=*9sequential/densenet169/conv3_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����("
	 �֬���"=*9sequential/densenet169/conv3_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����("
	 �ڬ���"=*9sequential/densenet169/conv3_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�ͦ�(���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����(���"��" *$$1"!*size:512 dest:0 async:1"�  ",`���(���"��" *$$1""8�"�  ",W��ζ(���"��" *$$1""8�"�  "�سܺ("
	 ������";*7sequential/densenet169/conv3_block8_1_bn/AssignNewValue"
 �������"  "�  "�����("
	 ������"=*9sequential/densenet169/conv3_block8_1_bn/AssignNewValue_1"
 �������"  "�  "�����("
	 ��̟��"5*1sequential/densenet169/conv3_block8_2_conv/Conv2D"
 �������"
*output" "*[200,32,4,4]"�  "��Ѕ�("
	 ������"5*1sequential/densenet169/conv3_block8_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����(ز�"��" *$$1""8�"�  "�����("
	 ������"5*1sequential/densenet169/conv3_block8_2_conv/Conv2D"
 �������"*temp" "
	 ������"5*1sequential/densenet169/conv3_block8_2_conv/Conv2D"
 �������"  "�  "�����("
	 ������"5*1sequential/densenet169/conv3_block8_2_conv/Conv2D"
 �������"  "�  "�����("
	 ������"5*1sequential/densenet169/conv3_block8_concat/concat"
 �������"
*output" "*
	 ��̟��"
  "  "�  "�����("
	 ������"=*9sequential/densenet169/conv3_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv3_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[384]"�  "�����("
	 �����"=*9sequential/densenet169/conv3_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[384]"�  "�Ў��("
	 �؄���"=*9sequential/densenet169/conv3_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[384]"�  "�����("
	 ��ī��"=*9sequential/densenet169/conv3_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[384]"�  "CU����(���"��" *$$1"!*size:1536 dest:0 async:1"�  "CU����(��"��" *$$1"!*size:1536 dest:0 async:1"�  ",`����(���"��" *$$1""8�"�  ",W����(���"��" *$$1""8�"�  "�؎��("
	 �̒���";*7sequential/densenet169/conv3_block9_0_bn/AssignNewValue"
 �������"  "�  "�����("
	 ������"=*9sequential/densenet169/conv3_block9_0_bn/AssignNewValue_1"
 �������"  "�  "��ŧ�("
	 ������"5*1sequential/densenet169/conv3_block9_1_conv/Conv2D"
 �������"
*output" "*
	 ��̟��"5*1sequential/densenet169/conv3_block9_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv3_block9_1_conv/Conv2D"
 �������"*temp" "	*[104]"�  ",S����)Ȃ�"��" *$$1""8�"�  ",T��Ջ)���"��" *$$1""8�"�  "��й�)"
	 ������"5*1sequential/densenet169/conv3_block9_1_conv/Conv2D"
 �������"  "�  "��Ȃ�)"
	 ��̟��"5*1sequential/densenet169/conv3_block9_1_conv/Conv2D"
 �������"  "�  "�и�)"
	 ��Ѐ��"=*9sequential/densenet169/conv3_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv3_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����)"
	 ������"=*9sequential/densenet169/conv3_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��꒒)"
	 ��Ŧ��"=*9sequential/densenet169/conv3_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����)"
	 ��Ŧ��"=*9sequential/densenet169/conv3_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�ۻ�)���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����)���"��" *$$1"!*size:512 dest:0 async:1"�  ",`���)���"��" *$$1""8�"�  ",W��ޠ)���"��" *$$1""8�"�  "�����)"
	 ����";*7sequential/densenet169/conv3_block9_1_bn/AssignNewValue"
 �������"  "�  "�ඡ�)"
	 ������"=*9sequential/densenet169/conv3_block9_1_bn/AssignNewValue_1"
 �������"  "�  "��)"
	 ��̟��"5*1sequential/densenet169/conv3_block9_2_conv/Conv2D"
 �������"
*output" "*[200,32,4,4]"�  "�����)"
	 ������"5*1sequential/densenet169/conv3_block9_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",Rп��)���"��" *$$1""8�"�  "�����)"
	 ������"5*1sequential/densenet169/conv3_block9_2_conv/Conv2D"
 �������"*temp" "
	 ������"5*1sequential/densenet169/conv3_block9_2_conv/Conv2D"
 �������"  "�  "����)"
	 ������"5*1sequential/densenet169/conv3_block9_2_conv/Conv2D"
 �������"  "�  "���Ҿ)"
	 ������"5*1sequential/densenet169/conv3_block9_concat/concat"
 �������"
*output" "*
	 ��̟��"
  "  "�  "�����)"
�Y�>��?" ����" ���" ���"
	 �����">*:sequential/densenet169/conv3_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "*
�Y�>��?" ����" �
	 ��ī��">*:sequential/densenet169/conv3_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[416]"�  "�����)"
�Y�>��?" ����" �
	 �̒���">*:sequential/densenet169/conv3_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[416]"�  "�����)"
�Y�>��?" ����" �
	 ��ī��">*:sequential/densenet169/conv3_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[416]"�  "�����)"
	 ��ī��">*:sequential/densenet169/conv3_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[416]"�  "CU����)���"��" *$$1"!*size:1664 dest:0 async:1"�  "CU�Ŏ�)���"��" *$$1"!*size:1664 dest:0 async:1"�  ",`����)���"��" *$$1""8�"�  ",W����)���"��" *$$1""8�"�  "�����)"
�Y�>��?" ����" �
	 ������"<*8sequential/densenet169/conv3_block10_0_bn/AssignNewValue"
 �������"  "�  "�����)"
�Y�>��?" ����" �
	 ������">*:sequential/densenet169/conv3_block10_0_bn/AssignNewValue_1"
 �������"  "�  "����)"
	 �췣��"6*2sequential/densenet169/conv3_block10_1_conv/Conv2D"
 �������"
*output" "*
	 ��̟��"6*2sequential/densenet169/conv3_block10_1_conv/Conv2D"
 �������"*temp" "*
	 ����"6*2sequential/densenet169/conv3_block10_1_conv/Conv2D"
 �������"*temp" "	*[104]"�  ",S����)ȣ�"��" *$$1""8�"�  ",T�ƅ�)���"��" *$$1""8�"�  "�����)"
	 ����"6*2sequential/densenet169/conv3_block10_1_conv/Conv2D"
 �������"  "�  "�І��)"
	 ��̟��"6*2sequential/densenet169/conv3_block10_1_conv/Conv2D"
 �������"  "�  "�����)"
	 �웤��">*:sequential/densenet169/conv3_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ����">*:sequential/densenet169/conv3_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����)"
	 ������">*:sequential/densenet169/conv3_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�Ȫ��)"
	 ��Ʀ��">*:sequential/densenet169/conv3_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����)"
	 ��Ʀ��">*:sequential/densenet169/conv3_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����)���"��" *$$1"!*size:512 dest:0 async:1"�  "BU�ʦ�)���"��" *$$1"!*size:512 dest:0 async:1"�  ",`��ۂ*�Ղ"��" *$$1""8�"�  ",W��ɇ*а�"��" *$$1""8�"�  "��ߍ�*"
	 ������"<*8sequential/densenet169/conv3_block10_1_bn/AssignNewValue"
 �������"  "�  "�����*"
	 ������">*:sequential/densenet169/conv3_block10_1_bn/AssignNewValue_1"
 �������"  "�  "�ؓ��*"
	 ��̟��"6*2sequential/densenet169/conv3_block10_2_conv/Conv2D"
 �������"
*output" "*[200,32,4,4]"�  "�����*"
	 ������"6*2sequential/densenet169/conv3_block10_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����*ػ�"��" *$$1""8�"�  "�ౝ�*"
	 �숥��"6*2sequential/densenet169/conv3_block10_2_conv/Conv2D"
 �������"*temp" "
	 �숥��"6*2sequential/densenet169/conv3_block10_2_conv/Conv2D"
 �������"  "�  "�ȭ��*"
	 ������"6*2sequential/densenet169/conv3_block10_2_conv/Conv2D"
 �������"  "�  "�طޥ*"
	 ������"6*2sequential/densenet169/conv3_block10_concat/concat"
 �������"
*output" "*
	 ��̟��"
  "  "�  "��̮*"
	 ��ݧ��">*:sequential/densenet169/conv3_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv3_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[448]"�  "���߯*"
	 ������">*:sequential/densenet169/conv3_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[448]"�  "�����*"
	 ������">*:sequential/densenet169/conv3_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[448]"�  "�����*"
	 ��ī��">*:sequential/densenet169/conv3_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[448]"�  "CUȹɱ*���"��" *$$1"!*size:1792 dest:0 async:1"�  "CU�÷�*�ׄ"��" *$$1"!*size:1792 dest:0 async:1"�  ",`؈��*���"��" *$$1""8�"�  ",W���*��"��" *$$1""8�"�  "�،��*"
	 �֓���"<*8sequential/densenet169/conv3_block11_0_bn/AssignNewValue"
 �������"  "�  "�Ȑ��*"
	 ������">*:sequential/densenet169/conv3_block11_0_bn/AssignNewValue_1"
 �������"  "�  "��ې�*"
	 �컪��"6*2sequential/densenet169/conv3_block11_1_conv/Conv2D"
 �������"
*output" "*
	 ��̟��"6*2sequential/densenet169/conv3_block11_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv3_block11_1_conv/Conv2D"
 �������"*temp" "	*[104]"�  ",S����*���"��" *$$1""8�"�  ",T���*���"��" *$$1""8�"�  "�����*"
	 ������"6*2sequential/densenet169/conv3_block11_1_conv/Conv2D"
 �������"  "�  "�����*"
	 ��̟��"6*2sequential/densenet169/conv3_block11_1_conv/Conv2D"
 �������"  "�  "�����*"
	 �쟫��">*:sequential/densenet169/conv3_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv3_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�آ��*"
	 ������">*:sequential/densenet169/conv3_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����*"
	 ��Ǧ��">*:sequential/densenet169/conv3_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����*"
	 ��Ǧ��">*:sequential/densenet169/conv3_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����*���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����*���"��" *$$1"!*size:512 dest:0 async:1"�  ",`Ѓ��*���"��" *$$1""8�"�  ",W���*���"��" *$$1""8�"�  "�����*"
	 ������"<*8sequential/densenet169/conv3_block11_1_bn/AssignNewValue"
 �������"  "�  "����*"
	 ������">*:sequential/densenet169/conv3_block11_1_bn/AssignNewValue_1"
 �������"  "�  "�����*"
	 ��̟��"6*2sequential/densenet169/conv3_block11_2_conv/Conv2D"
 �������"
*output" "*[200,32,4,4]"�  "����*"
	 �샬��"6*2sequential/densenet169/conv3_block11_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����*���"��" *$$1""8�"�  "����*"
	 �쌬��"6*2sequential/densenet169/conv3_block11_2_conv/Conv2D"
 �������"*temp" "
	 �쌬��"6*2sequential/densenet169/conv3_block11_2_conv/Conv2D"
 �������"  "�  "����+"
	 �샬��"6*2sequential/densenet169/conv3_block11_2_conv/Conv2D"
 �������"  "�  "���щ+"
	 �샬��"6*2sequential/densenet169/conv3_block11_concat/concat"
 �������"
*output" "*
	 ��̟��"
  "  "�  "�П��+"
	 ������">*:sequential/densenet169/conv3_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ī��">*:sequential/densenet169/conv3_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[480]"�  "��푓+"
	 �֓���">*:sequential/densenet169/conv3_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[480]"�  "��۾�+"
	 ��ū��">*:sequential/densenet169/conv3_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[480]"�  "�Б�+"
	 ��ū��">*:sequential/densenet169/conv3_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[480]"�  "CU���+���"��" *$$1"!*size:1920 dest:0 async:1"�  "CU����+�È"��" *$$1"!*size:1920 dest:0 async:1"�  ",`𼿝+���"��" *$$1""8�"�  ",W����+��"��" *$$1""8�"�  "�Ȁ��+"
	 ������"<*8sequential/densenet169/conv3_block12_0_bn/AssignNewValue"
 �������"  "�  "�����+"
	 ������">*:sequential/densenet169/conv3_block12_0_bn/AssignNewValue_1"
 �������"  "�  "��Ц�+"
	 �����"6*2sequential/densenet169/conv3_block12_1_conv/Conv2D"
 �������"
*output" "*
	 ��̟��"6*2sequential/densenet169/conv3_block12_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv3_block12_1_conv/Conv2D"
 �������"*temp" "	*[104]"�  ",S����+���"��" *$$1""8�"�  ",T����+���"��" *$$1""8�"�  "�����+"
	 ������"6*2sequential/densenet169/conv3_block12_1_conv/Conv2D"
 �������"  "�  "���½+"
	 ��̟��"6*2sequential/densenet169/conv3_block12_1_conv/Conv2D"
 �������"  "�  "��ȫ�+"
	 ��ղ��">*:sequential/densenet169/conv3_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv3_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����+"
	 ������">*:sequential/densenet169/conv3_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����+"
	 ��Ȧ��">*:sequential/densenet169/conv3_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����+"
	 ��Ȧ��">*:sequential/densenet169/conv3_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����+ؤ�"��" *$$1"!*size:512 dest:0 async:1"�  "BU����+؆�"��" *$$1"!*size:512 dest:0 async:1"�  ",`����+ؿ�"��" *$$1""8�"�  ",W����+���"��" *$$1""8�"�  "�衖�+"
	 �Ȕ���"<*8sequential/densenet169/conv3_block12_1_bn/AssignNewValue"
 �������"  "�  "��Ȩ�+"
	 �̔���">*:sequential/densenet169/conv3_block12_1_bn/AssignNewValue_1"
 �������"  "�  "�����+"
	 ��̟��"6*2sequential/densenet169/conv3_block12_2_conv/Conv2D"
 �������"
*output" "*[200,32,4,4]"�  "�����+"
	 �칳��"6*2sequential/densenet169/conv3_block12_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����+���"��" *$$1""8�"�  "�����+"
	 ��³��"6*2sequential/densenet169/conv3_block12_2_conv/Conv2D"
 �������"*temp" "
	 ��³��"6*2sequential/densenet169/conv3_block12_2_conv/Conv2D"
 �������"  "�  "�أ��+"
	 �칳��"6*2sequential/densenet169/conv3_block12_2_conv/Conv2D"
 �������"  "�  "�����+"
	 �칳��"6*2sequential/densenet169/conv3_block12_concat/concat"
 �������"
*output" "*
	 ��̟��"
  "  "�  "�����+"
	 ��ɶ��"4*0sequential/densenet169/pool3_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"4*0sequential/densenet169/pool3_bn/FusedBatchNormV3"
 �������"
*output" "	*[512]"�  "�����+"
	 �����"4*0sequential/densenet169/pool3_bn/FusedBatchNormV3"
 �������"
*output" "	*[512]"�  "�����+"
	 �����"4*0sequential/densenet169/pool3_bn/FusedBatchNormV3"
 �������"
*output" "	*[512]"�  "�����+"
	 ��ū��"4*0sequential/densenet169/pool3_bn/FusedBatchNormV3"
 �������"
*output" "	*[512]"�  "CU����+Ȗ�"��" *$$1"!*size:2048 dest:0 async:1"�  "CU����+���"��" *$$1"!*size:2048 dest:0 async:1"�  ",`�⯂,���"��" *$$1""8�"�  ",W����,�ˀ"��" *$$1""8�"�  "���ˋ,"
	 ����"2*.sequential/densenet169/pool3_bn/AssignNewValue"
 �������"  "�  "���č,"
	 ������"4*0sequential/densenet169/pool3_bn/AssignNewValue_1"
 �������"  "�  "���Ő,"
	 ��ٹ��",*(sequential/densenet169/pool3_conv/Conv2D"
 �������"
*output" "*
	 �졻��",*(sequential/densenet169/pool3_conv/Conv2D"
 �������"*temp" "*
	 �Ȕ���",*(sequential/densenet169/pool3_conv/Conv2D"
 �������"*temp" "	*[104]"�  ",S��Ĝ,�ڐ"��" *$$1""8�"�  ",T���,���"��" *$$1""8�"�  "�����,"
	 �Ȕ���",*(sequential/densenet169/pool3_conv/Conv2D"
 �������"  "�  "��,"
	 �졻��",*(sequential/densenet169/pool3_conv/Conv2D"
 �������"  "�  "�����,"
	 �졻��"-*)sequential/densenet169/pool3_pool/AvgPool"
 �������"
*output" "*
	 ��ٹ��"
  "  "�  "���ƭ,"
	 ��ٹ��"=*9sequential/densenet169/conv4_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �Ȕ���"=*9sequential/densenet169/conv4_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[256]"�  "�����,"
	 ��ɦ��"=*9sequential/densenet169/conv4_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[256]"�  "���,"
	 ��ʦ��"=*9sequential/densenet169/conv4_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[256]"�  "����,"
	 ��˦��"=*9sequential/densenet169/conv4_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[256]"�  "CU�ـ�,���"��" *$$1"!*size:1024 dest:0 async:1"�  "CU�ݬ�,�Ղ"��" *$$1"!*size:1024 dest:0 async:1"�  ",`���,���"��" *$$1""8�"�  ",W��ƾ,���"��" *$$1""8�"�  "�����,"
	 ������";*7sequential/densenet169/conv4_block1_0_bn/AssignNewValue"
 �������"  "�  "�؉��,"
	 ������"=*9sequential/densenet169/conv4_block1_0_bn/AssignNewValue_1"
 �������"  "�  "����,"
	 ��̟��"5*1sequential/densenet169/conv4_block1_1_conv/Conv2D"
 �������"
*output" "*
	 �싺��"5*1sequential/densenet169/conv4_block1_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv4_block1_1_conv/Conv2D"
 �������"*temp" "*[32]"�  ",S����,���"��" *$$1""8�"�  ",T����,���"��" *$$1""8�"�  "�茈�,"
	 ������"5*1sequential/densenet169/conv4_block1_1_conv/Conv2D"
 �������"  "�  "�����,"
	 �싺��"5*1sequential/densenet169/conv4_block1_1_conv/Conv2D"
 �������"  "�  "�����,"
	 �싺��"=*9sequential/densenet169/conv4_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv4_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����,"
	 ������"=*9sequential/densenet169/conv4_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����,"
	 ������"=*9sequential/densenet169/conv4_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����,"
	 ������"=*9sequential/densenet169/conv4_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����,���"��" *$$1"!*size:512 dest:0 async:1"�  "BU���,���"��" *$$1"!*size:512 dest:0 async:1"�  ",`���,���"��" *$$1""8�"�  ",W����,��"��" *$$1""8�"�  "��֥�,"
	 �ԕ���";*7sequential/densenet169/conv4_block1_1_bn/AssignNewValue"
 �������"  "�  "����,"
	 �ؕ���"=*9sequential/densenet169/conv4_block1_1_bn/AssignNewValue_1"
 �������"  "�  "�����,"
	 �줺��"5*1sequential/densenet169/conv4_block1_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����,"
	 ������"5*1sequential/densenet169/conv4_block1_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����,�Φ"��" *$$1""8�"�  "�����,"
	 ��ӻ��"5*1sequential/densenet169/conv4_block1_2_conv/Conv2D"
 �������"*temp" "
	 ��ӻ��"5*1sequential/densenet169/conv4_block1_2_conv/Conv2D"
 �������"  "�  "��ʘ�-"
	 ������"5*1sequential/densenet169/conv4_block1_2_conv/Conv2D"
 �������"  "�  "�؉ֈ-"
	 ������"5*1sequential/densenet169/conv4_block1_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "���ɑ-"
	 �����"=*9sequential/densenet169/conv4_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv4_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[288]"�  "���Ւ-"
"
	 ��ū��"=*9sequential/densenet169/conv4_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[288]"�  "�����-"
"
	 ��ū��"=*9sequential/densenet169/conv4_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[288]"�  "�����-"
"
	 ����"=*9sequential/densenet169/conv4_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[288]"�  "CU����-ȼ�"��" *$$1"!*size:1152 dest:0 async:1"�  "CU��Ș-ȗ�"��" *$$1"!*size:1152 dest:0 async:1"�  ",`�࡝-���"��" *$$1""8�"�  ",W�ɾ�-���"��" *$$1""8�"�  "�����-"
"
	 ��ë��";*7sequential/densenet169/conv4_block2_0_bn/AssignNewValue"
 �������"  "�  "�൛�-"
"
	 ��ë��"=*9sequential/densenet169/conv4_block2_0_bn/AssignNewValue_1"
 �������"  "�  "�Ȅ��-"
	 ��ӻ��"5*1sequential/densenet169/conv4_block2_1_conv/Conv2D"
 �������"
*output" "*
	 �����"5*1sequential/densenet169/conv4_block2_1_conv/Conv2D"
 �������"*temp" "*
	 �ԕ���"5*1sequential/densenet169/conv4_block2_1_conv/Conv2D"
 �������"*temp" "*[32]"�  ",S�Հ�-�͏"��" *$$1""8�"�  ",T����-���"��" *$$1""8�"�  "�����-"
	 �ԕ���"5*1sequential/densenet169/conv4_block2_1_conv/Conv2D"
 �������"  "�  "�Б˾-"
	 �����"5*1sequential/densenet169/conv4_block2_1_conv/Conv2D"
 �������"  "�  "�賷�-"
	 �����"=*9sequential/densenet169/conv4_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �ԕ���"=*9sequential/densenet169/conv4_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�л��-"
	 �ؕ���"=*9sequential/densenet169/conv4_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����-"
	 ��̦��"=*9sequential/densenet169/conv4_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����-"
	 ��̦��"=*9sequential/densenet169/conv4_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BUȉ��-���"��" *$$1"!*size:512 dest:0 async:1"�  "BU��-���"��" *$$1"!*size:512 dest:0 async:1"�  ",`���-���"��" *$$1""8�"�  ",W耘�-���"��" *$$1""8�"�  "�����-"
	 �ܕ���";*7sequential/densenet169/conv4_block2_1_bn/AssignNewValue"
 �������"  "�  "�����-"
	 ������"=*9sequential/densenet169/conv4_block2_1_bn/AssignNewValue_1"
 �������"  "�  "����-"
	 �줺��"5*1sequential/densenet169/conv4_block2_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����-"
	 �셼��"5*1sequential/densenet169/conv4_block2_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",RЕ��-���"��" *$$1""8�"�  "�����-"
	 �쎼��"5*1sequential/densenet169/conv4_block2_2_conv/Conv2D"
 �������"*temp" "
	 �쎼��"5*1sequential/densenet169/conv4_block2_2_conv/Conv2D"
 �������"  "�  "�����-"
	 �셼��"5*1sequential/densenet169/conv4_block2_2_conv/Conv2D"
 �������"  "�  "�Ы��-"
	 �셼��"5*1sequential/densenet169/conv4_block2_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "�����-"
	 ��ļ��"=*9sequential/densenet169/conv4_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "*
" �"
	 ������"=*9sequential/densenet169/conv4_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[320]"�  "�����-"
" �
"
	 ��ƫ��"=*9sequential/densenet169/conv4_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[320]"�  "�����-"
" �
"
	 ��ƫ��"=*9sequential/densenet169/conv4_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[320]"�  "��."
" �"
	 ��ƫ��"=*9sequential/densenet169/conv4_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[320]"�  "CU����.���"��" *$$1"!*size:1280 dest:0 async:1"�  "CU�؟�.���"��" *$$1"!*size:1280 dest:0 async:1"�  ",`����.���"��" *$$1""8�"�  ",Wأُ.���"��" *$$1""8�"�  "�����."
" �
"
	 ������";*7sequential/densenet169/conv4_block3_0_bn/AssignNewValue"
 �������"  "�  "��ؔ."
" �"
	 �ʛ���"=*9sequential/densenet169/conv4_block3_0_bn/AssignNewValue_1"
 �������"  "�  "����."
	 �삽��"5*1sequential/densenet169/conv4_block3_1_conv/Conv2D"
 �������"
*output" "*
" ��
"
	 �웽��"5*1sequential/densenet169/conv4_block3_1_conv/Conv2D"
 �������"*temp" "*
	 �ܕ���"5*1sequential/densenet169/conv4_block3_1_conv/Conv2D"
 �������"*temp" "*[32]"�  ",S����.���"��" *$$1""8�"�  ",T����.���"��" *$$1""8�"�  "�����."
	 �ܕ���"5*1sequential/densenet169/conv4_block3_1_conv/Conv2D"
 �������"  "�  "�гɪ."
" ��
"
	 �웽��"5*1sequential/densenet169/conv4_block3_1_conv/Conv2D"
 �������"  "�  "�����."
	 �웽��"=*9sequential/densenet169/conv4_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �ܕ���"=*9sequential/densenet169/conv4_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����."
	 ������"=*9sequential/densenet169/conv4_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����."
	 �����"=*9sequential/densenet169/conv4_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���."
	 �����"=*9sequential/densenet169/conv4_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����.���"��" *$$1"!*size:512 dest:0 async:1"�  "BU���.���"��" *$$1"!*size:512 dest:0 async:1"�  ",`�࠷.���"��" *$$1""8�"�  ",W����.���"��" *$$1""8�"�  "���ҿ."
	 ������";*7sequential/densenet169/conv4_block3_1_bn/AssignNewValue"
 �������"  "�  "��ί�."
	 ������"=*9sequential/densenet169/conv4_block3_1_bn/AssignNewValue_1"
 �������"  "�  "��ܴ�."
	 �줺��"5*1sequential/densenet169/conv4_block3_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����."
	 �촽��"5*1sequential/densenet169/conv4_block3_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R�۾�.Ў�"��" *$$1""8�"�  "�����."
	 �콽��"5*1sequential/densenet169/conv4_block3_2_conv/Conv2D"
 �������"*temp" "
	 �콽��"5*1sequential/densenet169/conv4_block3_2_conv/Conv2D"
 �������"  "�  "�����."
	 �촽��"5*1sequential/densenet169/conv4_block3_2_conv/Conv2D"
 �������"  "�  "�ؓ��."
e��k�?" ����" ��D" ��D"
	 �촽��"5*1sequential/densenet169/conv4_block3_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "�ؼ��."
	 ������"=*9sequential/densenet169/conv4_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv4_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[352]"�  "�����."
	 �̛���"=*9sequential/densenet169/conv4_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[352]"�  "�����."
	 ������"=*9sequential/densenet169/conv4_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[352]"�  "�����."
	 ������"=*9sequential/densenet169/conv4_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[352]"�  "CU����.���"��" *$$1"!*size:1408 dest:0 async:1"�  "CU����.��"��" *$$1"!*size:1408 dest:0 async:1"�  ",`����.��"��" *$$1""8�"�  ",Wк��.���"��" *$$1""8�"�  "�����."
	 ��ë��";*7sequential/densenet169/conv4_block4_0_bn/AssignNewValue"
 �������"  "�  "�����."
	 ��ë��"=*9sequential/densenet169/conv4_block4_0_bn/AssignNewValue_1"
 �������"  "�  "�����."
	 ������"5*1sequential/densenet169/conv4_block4_1_conv/Conv2D"
 �������"
*output" "*
	 ��׾��"5*1sequential/densenet169/conv4_block4_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv4_block4_1_conv/Conv2D"
 �������"*temp" "*[32]"�  ",S���/���"��" *$$1""8�"�  ",T��ڌ/���"��" *$$1""8�"�  "�زˏ/"
	 ������"5*1sequential/densenet169/conv4_block4_1_conv/Conv2D"
 �������"  "�  "��Ȥ�/"
	 ��׾��"5*1sequential/densenet169/conv4_block4_1_conv/Conv2D"
 �������"  "�  "�����/"
	 ��׾��"=*9sequential/densenet169/conv4_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv4_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��愓/"
	 ������"=*9sequential/densenet169/conv4_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�ȗ��/"
	 �����"=*9sequential/densenet169/conv4_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��Փ/"
	 �����"=*9sequential/densenet169/conv4_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU��ڔ/���"��" *$$1"!*size:512 dest:0 async:1"�  "BU���/���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����/���"��" *$$1""8�"�  ",W����/���"��" *$$1""8�"�  "���٪/"
	 ������";*7sequential/densenet169/conv4_block4_1_bn/AssignNewValue"
 �������"  "�  "�����/"
	 ������"=*9sequential/densenet169/conv4_block4_1_bn/AssignNewValue_1"
 �������"  "�  "�����/"
	 �줺��"5*1sequential/densenet169/conv4_block4_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����/"
	 �����"5*1sequential/densenet169/conv4_block4_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����/���"��" *$$1""8�"�  "�ؾҵ/"
	 ������"5*1sequential/densenet169/conv4_block4_2_conv/Conv2D"
 �������"*temp" "
	 ������"5*1sequential/densenet169/conv4_block4_2_conv/Conv2D"
 �������"  "�  "�����/"
	 �����"5*1sequential/densenet169/conv4_block4_2_conv/Conv2D"
 �������"  "�  "�����/"
	 �����"5*1sequential/densenet169/conv4_block4_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "����/"
	 ������"=*9sequential/densenet169/conv4_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv4_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[384]"�  "�����/"
	 ��ƫ��"=*9sequential/densenet169/conv4_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[384]"�  "�����/"
	 ��ƫ��"=*9sequential/densenet169/conv4_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[384]"�  "�����/"
	 ��ƫ��"=*9sequential/densenet169/conv4_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[384]"�  "CU����/���"��" *$$1"!*size:1536 dest:0 async:1"�  "CU����/�˕"��" *$$1"!*size:1536 dest:0 async:1"�  ",`����/���"��" *$$1""8�"�  ",W����/���"��" *$$1""8�"�  "�����/"
	 ����";*7sequential/densenet169/conv4_block5_0_bn/AssignNewValue"
 �������"  "�  "��ܣ�/"
	 �·���"=*9sequential/densenet169/conv4_block5_0_bn/AssignNewValue_1"
 �������"  "�  "�����/"
	 ������"5*1sequential/densenet169/conv4_block5_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv4_block5_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv4_block5_1_conv/Conv2D"
 �������"*temp" "*[32]"�  ",S����/���"��" *$$1""8�"�  ",T����/���"��" *$$1""8�"�  "�Ȇ��/"
	 ������"5*1sequential/densenet169/conv4_block5_1_conv/Conv2D"
 �������"  "�  "�����/"
	 ������"5*1sequential/densenet169/conv4_block5_1_conv/Conv2D"
 �������"  "�  "��؋�/"
	 ������"=*9sequential/densenet169/conv4_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv4_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����/"
	 ������"=*9sequential/densenet169/conv4_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ѯ�/"
	 �����"=*9sequential/densenet169/conv4_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����/"
	 �����"=*9sequential/densenet169/conv4_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����/�ٍ"��" *$$1"!*size:512 dest:0 async:1"�  "BU����0ȋ�"��" *$$1"!*size:512 dest:0 async:1"�  ",`���0���"��" *$$1""8�"�  ",WЖ�0���"��" *$$1""8�"�  "�����0"
	 �⇥��";*7sequential/densenet169/conv4_block5_1_bn/AssignNewValue"
 �������"  "�  "����0"
	 �懥��"=*9sequential/densenet169/conv4_block5_1_bn/AssignNewValue_1"
 �������"  "�  "�����0"
	 �줺��"5*1sequential/densenet169/conv4_block5_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����0"
	 ������"5*1sequential/densenet169/conv4_block5_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",RЌ��0�ή"��" *$$1""8�"�  "���ŝ0"
	 ������"5*1sequential/densenet169/conv4_block5_2_conv/Conv2D"
 �������"*temp" "
	 ������"5*1sequential/densenet169/conv4_block5_2_conv/Conv2D"
 �������"  "�  "����0"
	 ������"5*1sequential/densenet169/conv4_block5_2_conv/Conv2D"
 �������"  "�  "�ؘǬ0"
	 ������"5*1sequential/densenet169/conv4_block5_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "��ӵ0"
	 �̉���"=*9sequential/densenet169/conv4_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ����"=*9sequential/densenet169/conv4_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[416]"�  "��ض0"
	 ��ƫ��"=*9sequential/densenet169/conv4_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[416]"�  "�����0"
	 ��ǫ��"=*9sequential/densenet169/conv4_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[416]"�  "�����0"
	 �����"=*9sequential/densenet169/conv4_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[416]"�  "CU��Ӹ0д�"��" *$$1"!*size:1664 dest:0 async:1"�  "CU����0���"��" *$$1"!*size:1664 dest:0 async:1"�  ",`�Ν�0���"��" *$$1""8�"�  ",W����0��"��" *$$1""8�"�  "�����0"
	 ������";*7sequential/densenet169/conv4_block6_0_bn/AssignNewValue"
 �������"  "�  "�����0"
	 ��ī��"=*9sequential/densenet169/conv4_block6_0_bn/AssignNewValue_1"
 �������"  "�  "�����0"
	 ������"5*1sequential/densenet169/conv4_block6_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv4_block6_1_conv/Conv2D"
 �������"*temp" "*
	 �⇥��"5*1sequential/densenet169/conv4_block6_1_conv/Conv2D"
 �������"*temp" "*[32]"�  ",S����0���"��" *$$1""8�"�  ",T���0���"��" *$$1""8�"�  "��̺�0"
	 �⇥��"5*1sequential/densenet169/conv4_block6_1_conv/Conv2D"
 �������"  "�  "�����0"
	 ������"5*1sequential/densenet169/conv4_block6_1_conv/Conv2D"
 �������"  "�  "����0"
	 ������"=*9sequential/densenet169/conv4_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �⇥��"=*9sequential/densenet169/conv4_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����0"
	 �懥��"=*9sequential/densenet169/conv4_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����0"
	 ������"=*9sequential/densenet169/conv4_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����0"
	 ������"=*9sequential/densenet169/conv4_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����0���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����0���"��" *$$1"!*size:512 dest:0 async:1"�  ",`���0���"��" *$$1""8�"�  ",W����0���"��" *$$1""8�"�  "�����1"
	 ������";*7sequential/densenet169/conv4_block6_1_bn/AssignNewValue"
 �������"  "�  "��⦃1"
	 ������"=*9sequential/densenet169/conv4_block6_1_bn/AssignNewValue_1"
 �������"  "�  "�����1"
	 �줺��"5*1sequential/densenet169/conv4_block6_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "���1"
	 �����"5*1sequential/densenet169/conv4_block6_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R���1���"��" *$$1""8�"�  "��윍1"
	 �����"5*1sequential/densenet169/conv4_block6_2_conv/Conv2D"
 �������"*temp" "
	 �����"5*1sequential/densenet169/conv4_block6_2_conv/Conv2D"
 �������"  "�  "��ƙ1"
	 �����"5*1sequential/densenet169/conv4_block6_2_conv/Conv2D"
 �������"  "�  "�����1"
	 �����"5*1sequential/densenet169/conv4_block6_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "�؆ܤ1"
	 ������"=*9sequential/densenet169/conv4_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv4_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[448]"�  "���ԥ1"
	 �����"=*9sequential/densenet169/conv4_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[448]"�  "�ࣆ�1"
	 ��ǫ��"=*9sequential/densenet169/conv4_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[448]"�  "�����1"
	 ��ǫ��"=*9sequential/densenet169/conv4_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[448]"�  "CU����1���"��" *$$1"!*size:1792 dest:0 async:1"�  "CU�흫1���"��" *$$1"!*size:1792 dest:0 async:1"�  ",`��İ1���"��" *$$1""8�"�  ",W�ų�1���"��" *$$1""8�"�  "���κ1"
	 �Έ���";*7sequential/densenet169/conv4_block7_0_bn/AssignNewValue"
 �������"  "�  "���ֻ1"
	 ��ī��"=*9sequential/densenet169/conv4_block7_0_bn/AssignNewValue_1"
 �������"  "�  "����1"
	 �����"5*1sequential/densenet169/conv4_block7_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv4_block7_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv4_block7_1_conv/Conv2D"
 �������"*temp" "*[32]"�  ",S����1�Ό"��" *$$1""8�"�  ",T����1���"��" *$$1""8�"�  "����1"
	 ������"5*1sequential/densenet169/conv4_block7_1_conv/Conv2D"
 �������"  "�  "�����1"
	 ������"5*1sequential/densenet169/conv4_block7_1_conv/Conv2D"
 �������"  "�  "�����1"
	 ������"=*9sequential/densenet169/conv4_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv4_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����1"
	 ������"=*9sequential/densenet169/conv4_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����1"
	 ��ī��"=*9sequential/densenet169/conv4_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����1"
	 ��ī��"=*9sequential/densenet169/conv4_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����1���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����1�׉"��" *$$1"!*size:512 dest:0 async:1"�  ",`Ȟ��1���"��" *$$1""8�"�  ",W����1���"��" *$$1""8�"�  "�����1"
	 �ꈥ��";*7sequential/densenet169/conv4_block7_1_bn/AssignNewValue"
 �������"  "�  "�����1"
	 ���"=*9sequential/densenet169/conv4_block7_1_bn/AssignNewValue_1"
 �������"  "�  "����1"
	 �줺��"5*1sequential/densenet169/conv4_block7_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "��ج�1"
	 ������"5*1sequential/densenet169/conv4_block7_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����1���"��" *$$1""8�"�  "�Ȍ��1"
	 ������"5*1sequential/densenet169/conv4_block7_2_conv/Conv2D"
 �������"*temp" "
	 ������"5*1sequential/densenet169/conv4_block7_2_conv/Conv2D"
 �������"  "�  "�����1"
	 ������"5*1sequential/densenet169/conv4_block7_2_conv/Conv2D"
 �������"  "�  "�؂�2"
	 ������"5*1sequential/densenet169/conv4_block7_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "�����2"
	 ������"=*9sequential/densenet169/conv4_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ǫ��"=*9sequential/densenet169/conv4_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[480]"�  "���݋2"
	 �Έ���"=*9sequential/densenet169/conv4_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[480]"�  "�����2"
	 ��ǫ��"=*9sequential/densenet169/conv4_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[480]"�  "�����2"
	 ��ȫ��"=*9sequential/densenet169/conv4_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[480]"�  "CU����2Є�"��" *$$1"!*size:1920 dest:0 async:1"�  "CU؅��2���"��" *$$1"!*size:1920 dest:0 async:1"�  ",`����2���"��" *$$1""8�"�  ",W��ʜ2�ن"��" *$$1""8�"�  "��죠2"
	 ������";*7sequential/densenet169/conv4_block8_0_bn/AssignNewValue"
 �������"  "�  "����2"
	 ��ī��"=*9sequential/densenet169/conv4_block8_0_bn/AssignNewValue_1"
 �������"  "�  "��ܤ2"
	 ������"5*1sequential/densenet169/conv4_block8_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv4_block8_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv4_block8_1_conv/Conv2D"
 �������"  "�  "���ڻ2"
	 ������"=*9sequential/densenet169/conv4_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ī��"=*9sequential/densenet169/conv4_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���ټ2"
	 �ꈥ��"=*9sequential/densenet169/conv4_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��﻽2"
	 ���"=*9sequential/densenet169/conv4_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����2"
	 ��ī��"=*9sequential/densenet169/conv4_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����2���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����2���"��" *$$1"!*size:512 dest:0 async:1"�  ",`ا��2���"��" *$$1""8�"�  ",W����2���"��" *$$1""8�"�  "�����2"
	 ������";*7sequential/densenet169/conv4_block8_1_bn/AssignNewValue"
 �������"  "�  "�Ѓ��2"
	 ������"=*9sequential/densenet169/conv4_block8_1_bn/AssignNewValue_1"
 �������"  "�  "����2"
	 �줺��"5*1sequential/densenet169/conv4_block8_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����2"
	 ������"5*1sequential/densenet169/conv4_block8_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����2���"��" *$$1""8�"�  "�����2"
	 ������"5*1sequential/densenet169/conv4_block8_2_conv/Conv2D"
 �������"*temp" "
	 ������"5*1sequential/densenet169/conv4_block8_2_conv/Conv2D"
 �������"  "�  "�Ф��2"
	 ������"5*1sequential/densenet169/conv4_block8_2_conv/Conv2D"
 �������"  "�  "�Ȫ��2"
	 ������"5*1sequential/densenet169/conv4_block8_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "�����2"
	 ������"=*9sequential/densenet169/conv4_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv4_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[512]"�  "�����2"
	 ��ׄ��"=*9sequential/densenet169/conv4_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[512]"�  "�����2"
	 ��ׄ��"=*9sequential/densenet169/conv4_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[512]"�  "��Ä�2"
	 ��؄��"=*9sequential/densenet169/conv4_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[512]"�  "CU�ޒ�2��"��" *$$1"!*size:2048 dest:0 async:1"�  "CU�̄�2ؾ�"��" *$$1"!*size:2048 dest:0 async:1"�  ",`���3��"��" *$$1""8�"�  ",W��Ն3���"��" *$$1""8�"�  "�����3"
	 �䏥��";*7sequential/densenet169/conv4_block9_0_bn/AssignNewValue"
 �������"  "�  "�����3"
	 ��ū��"=*9sequential/densenet169/conv4_block9_0_bn/AssignNewValue_1"
 �������"  "�  "�঻�3"
	 ������"5*1sequential/densenet169/conv4_block9_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv4_block9_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv4_block9_1_conv/Conv2D"
 �������"*temp" "*[32]"�  ",S����3��"��" *$$1""8�"�  ",T࣌�3��"��" *$$1""8�"�  "�����3"
	 ������"5*1sequential/densenet169/conv4_block9_1_conv/Conv2D"
 �������"  "�  "�����3"
	 ������"5*1sequential/densenet169/conv4_block9_1_conv/Conv2D"
 �������"  "�  "�����3"
	 ������"=*9sequential/densenet169/conv4_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv4_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����3"
	 ������"=*9sequential/densenet169/conv4_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�ا��3"
	 ��ī��"=*9sequential/densenet169/conv4_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����3"
	 ��ī��"=*9sequential/densenet169/conv4_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����3���"��" *$$1"!*size:512 dest:0 async:1"�  "BU���3���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����3���"��" *$$1""8�"�  ",W��˳3���"��" *$$1""8�"�  "�胬�3"
	 ������";*7sequential/densenet169/conv4_block9_1_bn/AssignNewValue"
 �������"  "�  "���θ3"
	 ������"=*9sequential/densenet169/conv4_block9_1_bn/AssignNewValue_1"
 �������"  "�  "���ƺ3"
	 �줺��"5*1sequential/densenet169/conv4_block9_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "���ֻ3"
)?�?" ����" ��	" ��	"
	 ������"5*1sequential/densenet169/conv4_block9_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R��ϼ3��"��" *$$1""8�"�  "�����3"
	 ������"5*1sequential/densenet169/conv4_block9_2_conv/Conv2D"
 �������"*temp" "
)?�?" ����" ���" ���"
	 ������"5*1sequential/densenet169/conv4_block9_2_conv/Conv2D"
 �������"  "�  "�����3"
	 ������"5*1sequential/densenet169/conv4_block9_2_conv/Conv2D"
 �������"  "�  "�����3"
	 ������"5*1sequential/densenet169/conv4_block9_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "�����3"
	 �̿���">*:sequential/densenet169/conv4_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �䏥��">*:sequential/densenet169/conv4_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[544]"�  "�����3"
	 ��؄��">*:sequential/densenet169/conv4_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[544]"�  "��̐�3"
	 �ꮑ��">*:sequential/densenet169/conv4_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[544]"�  "�����3"
	 ������">*:sequential/densenet169/conv4_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[544]"�  "CU����3Ȩ�"��" *$$1"!*size:2176 dest:0 async:1"�  "CU�״�3���"��" *$$1"!*size:2176 dest:0 async:1"�  ",`����3���"��" *$$1""8�"�  ",W����3ت�"��" *$$1""8�"�  "�����3"
	 ������"<*8sequential/densenet169/conv4_block10_0_bn/AssignNewValue"
 �������"  "�  "�����3"
	 ��ū��">*:sequential/densenet169/conv4_block10_0_bn/AssignNewValue_1"
 �������"  "�  "�Ȟ��3"
	 �����"6*2sequential/densenet169/conv4_block10_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block10_1_conv/Conv2D"
 �������"*temp" "*
	 ��ī��"6*2sequential/densenet169/conv4_block10_1_conv/Conv2D"
 �������"*temp" "*[32]"�  ",S؅��3�ߑ"��" *$$1""8�"�  ",T�ņ�4���"��" *$$1""8�"�  "�����4"
	 ��ī��"6*2sequential/densenet169/conv4_block10_1_conv/Conv2D"
 �������"  "�  "���4"
	 ������"6*2sequential/densenet169/conv4_block10_1_conv/Conv2D"
 �������"  "�  "���4"
	 ������">*:sequential/densenet169/conv4_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ī��">*:sequential/densenet169/conv4_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����4"
	 ������">*:sequential/densenet169/conv4_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�й��4"
	 ������">*:sequential/densenet169/conv4_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��«�4"
	 ��ū��">*:sequential/densenet169/conv4_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����4ȩ�"��" *$$1"!*size:512 dest:0 async:1"�  "BU�ǀ�4���"��" *$$1"!*size:512 dest:0 async:1"�  ",`��ʒ4���"��" *$$1""8�"�  ",Wȶ��4���"��" *$$1""8�"�  "�����4"
	 ������"<*8sequential/densenet169/conv4_block10_1_bn/AssignNewValue"
 �������"  "�  "�軎�4"
	 ������">*:sequential/densenet169/conv4_block10_1_bn/AssignNewValue_1"
 �������"  "�  "�����4"
	 �줺��"6*2sequential/densenet169/conv4_block10_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����4"
	 ������"6*2sequential/densenet169/conv4_block10_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R�酠4�ڝ"��" *$$1""8�"�  "�����4"
	 ������"6*2sequential/densenet169/conv4_block10_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block10_2_conv/Conv2D"
 �������"  "�  "�����4"
	 ������"6*2sequential/densenet169/conv4_block10_2_conv/Conv2D"
 �������"  "�  "���޴4"
	 ������"6*2sequential/densenet169/conv4_block10_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "���ν4"
	 ������">*:sequential/densenet169/conv4_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ū��">*:sequential/densenet169/conv4_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[576]"�  "�ȰȾ4"
	 ������">*:sequential/densenet169/conv4_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[576]"�  "�����4"
	 ������">*:sequential/densenet169/conv4_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[576]"�  "�Т��4"
	 ������">*:sequential/densenet169/conv4_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[576]"�  "CU����4���"��" *$$1"!*size:2304 dest:0 async:1"�  "CU�ŷ�4���"��" *$$1"!*size:2304 dest:0 async:1"�  ",`诈�4��"��" *$$1""8�"�  ",W����4���"��" *$$1""8�"�  "�����4"
	 �ެ���"<*8sequential/densenet169/conv4_block11_0_bn/AssignNewValue"
 �������"  "�  "����4"
	 ��ū��">*:sequential/densenet169/conv4_block11_0_bn/AssignNewValue_1"
 �������"  "�  "�����4"
	 �����"6*2sequential/densenet169/conv4_block11_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block11_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv4_block11_1_conv/Conv2D"
 �������"*temp" "*[32]"�  ",S����4�ב"��" *$$1""8�"�  ",T����4���"��" *$$1""8�"�  "�����4"
	 ������"6*2sequential/densenet169/conv4_block11_1_conv/Conv2D"
 �������"  "�  "�����4"
	 ������"6*2sequential/densenet169/conv4_block11_1_conv/Conv2D"
 �������"  "�  "�Є��4"
	 ������">*:sequential/densenet169/conv4_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv4_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����4"
	 ������">*:sequential/densenet169/conv4_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����4"
	 ��ū��">*:sequential/densenet169/conv4_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��Ѹ�4"
	 ��ū��">*:sequential/densenet169/conv4_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����4���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����4���"��" *$$1"!*size:512 dest:0 async:1"�  ",`�˩�4���"��" *$$1""8�"�  ",W����4���"��" *$$1""8�"�  "�����4"
	 ������"<*8sequential/densenet169/conv4_block11_1_bn/AssignNewValue"
 �������"  "�  "�����5"
	 ������">*:sequential/densenet169/conv4_block11_1_bn/AssignNewValue_1"
 �������"  "�  "���5"
	 �줺��"6*2sequential/densenet169/conv4_block11_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�Ȍ��5"
	 ������"6*2sequential/densenet169/conv4_block11_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����5��"��" *$$1""8�"�  "�ؼ��5"
	 ������"6*2sequential/densenet169/conv4_block11_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block11_2_conv/Conv2D"
 �������"  "�  "��ޫ�5"
	 ������"6*2sequential/densenet169/conv4_block11_2_conv/Conv2D"
 �������"  "�  "���̘5"
	 ������"6*2sequential/densenet169/conv4_block11_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "����5"
	 ������">*:sequential/densenet169/conv4_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �ެ���">*:sequential/densenet169/conv4_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[608]"�  "����5"
	 ��ȫ��">*:sequential/densenet169/conv4_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[608]"�  "�Ȱ��5"
	 ��ȫ��">*:sequential/densenet169/conv4_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[608]"�  "����5"
	 �����">*:sequential/densenet169/conv4_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[608]"�  "CU����5��"��" *$$1"!*size:2432 dest:0 async:1"�  "CUІ��5���"��" *$$1"!*size:2432 dest:0 async:1"�  ",`�ō�5���"��" *$$1""8�"�  ",W����5���"��" *$$1""8�"�  "�̷ؙ5"
	 ������"<*8sequential/densenet169/conv4_block12_0_bn/AssignNewValue"
 �������"  "�  "���ո5"
	 ��ƫ��">*:sequential/densenet169/conv4_block12_0_bn/AssignNewValue_1"
 �������"  "�  "���»5"
	 ������"6*2sequential/densenet169/conv4_block12_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block12_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv4_block12_1_conv/Conv2D"
 �������"  "�  "����5"
	 ������">*:sequential/densenet169/conv4_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ū��">*:sequential/densenet169/conv4_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�Ș��5"
	 ������">*:sequential/densenet169/conv4_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��Ź�5"
	 ������">*:sequential/densenet169/conv4_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����5"
	 ��ū��">*:sequential/densenet169/conv4_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����5���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����5��"��" *$$1"!*size:512 dest:0 async:1"�  ",`У��5Ȭ�"��" *$$1""8�"�  ",W���5���"��" *$$1""8�"�  "�����5"
	 ����"<*8sequential/densenet169/conv4_block12_1_bn/AssignNewValue"
 �������"  "�  "�����5"
	 ������">*:sequential/densenet169/conv4_block12_1_bn/AssignNewValue_1"
 �������"  "�  "�����5"
	 �줺��"6*2sequential/densenet169/conv4_block12_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����5"
	 ������"6*2sequential/densenet169/conv4_block12_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����5���"��" *$$1""8�"�  "��ת�5"
	 ������"6*2sequential/densenet169/conv4_block12_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block12_2_conv/Conv2D"
 �������"  "�  "�����5"
	 ������"6*2sequential/densenet169/conv4_block12_2_conv/Conv2D"
 �������"  "�  "��֬�5"
	 ������"6*2sequential/densenet169/conv4_block12_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "�����6"
	 ������">*:sequential/densenet169/conv4_block13_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ƫ��">*:sequential/densenet169/conv4_block13_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[640]"�  "��э�6"
	 ������">*:sequential/densenet169/conv4_block13_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[640]"�  "�����6"
	 �����">*:sequential/densenet169/conv4_block13_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[640]"�  "����6"
	 �����">*:sequential/densenet169/conv4_block13_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[640]"�  "CU���6���"��" *$$1"!*size:2560 dest:0 async:1"�  "CU���6���"��" *$$1"!*size:2560 dest:0 async:1"�  ",`��Ð6���"��" *$$1""8�"�  ",W�ێ�6��"��" *$$1""8�"�  "�����6"
	 ������"<*8sequential/densenet169/conv4_block13_0_bn/AssignNewValue"
 �������"  "�  "��ŵ�6"
	 ��ƫ��">*:sequential/densenet169/conv4_block13_0_bn/AssignNewValue_1"
 �������"  "�  "��٨�6"
	 ������"6*2sequential/densenet169/conv4_block13_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block13_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv4_block13_1_conv/Conv2D"
 �������"  "�  "��ڄ�6"
	 ������">*:sequential/densenet169/conv4_block13_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ����">*:sequential/densenet169/conv4_block13_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���α6"
	 ������">*:sequential/densenet169/conv4_block13_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����6"
	 ��ū��">*:sequential/densenet169/conv4_block13_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����6"
	 ��ū��">*:sequential/densenet169/conv4_block13_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����6���"��" *$$1"!*size:512 dest:0 async:1"�  "BU��6��"��" *$$1"!*size:512 dest:0 async:1"�  ",`���6��"��" *$$1""8�"�  ",WО��6��"��" *$$1""8�"�  "���6"
	 �؊���"<*8sequential/densenet169/conv4_block13_1_bn/AssignNewValue"
 �������"  "�  "�Э��6"
	 �܊���">*:sequential/densenet169/conv4_block13_1_bn/AssignNewValue_1"
 �������"  "�  "�����6"
	 �줺��"6*2sequential/densenet169/conv4_block13_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����6"
	 ������"6*2sequential/densenet169/conv4_block13_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����6��"��" *$$1""8�"�  "��ق�6"
�?" ����" ���" ���"
	 ������"6*2sequential/densenet169/conv4_block13_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block13_2_conv/Conv2D"
 �������"  "�  "��؅�6"
	 ������"6*2sequential/densenet169/conv4_block13_2_conv/Conv2D"
 �������"  "�  "�����6"
	 ������"6*2sequential/densenet169/conv4_block13_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "��݌�6"
	 �̽���">*:sequential/densenet169/conv4_block14_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����">*:sequential/densenet169/conv4_block14_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[672]"�  "���6"
	 ������">*:sequential/densenet169/conv4_block14_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[672]"�  "�����6"
	 ������">*:sequential/densenet169/conv4_block14_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[672]"�  "�����6"
	 ������">*:sequential/densenet169/conv4_block14_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[672]"�  "CU����6���"��" *$$1"!*size:2688 dest:0 async:1"�  "CU����6Ⱦ�"��" *$$1"!*size:2688 dest:0 async:1"�  ",`Ч��6�ł"��" *$$1""8�"�  ",W����6���"��" *$$1""8�"�  "����6"
	 ������"<*8sequential/densenet169/conv4_block14_0_bn/AssignNewValue"
 �������"  "�  "��̖�6"
	 ��ǫ��">*:sequential/densenet169/conv4_block14_0_bn/AssignNewValue_1"
 �������"  "�  "��傁7"
	 ������"6*2sequential/densenet169/conv4_block14_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block14_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv4_block14_1_conv/Conv2D"
 �������"  "�  "��ے7"
	 ������">*:sequential/densenet169/conv4_block14_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ū��">*:sequential/densenet169/conv4_block14_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���ړ7"
	 �؊���">*:sequential/densenet169/conv4_block14_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����7"
	 �܊���">*:sequential/densenet169/conv4_block14_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����7"
	 ��ƫ��">*:sequential/densenet169/conv4_block14_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU���7�΁"��" *$$1"!*size:512 dest:0 async:1"�  "BU����7谖"��" *$$1"!*size:512 dest:0 async:1"�  ",`�π�7ȩ�"��" *$$1""8�"�  ",W����7���"��" *$$1""8�"�  "��ݧ7"
	 �ڋ���"<*8sequential/densenet169/conv4_block14_1_bn/AssignNewValue"
 �������"  "�  "����7"
	 �ދ���">*:sequential/densenet169/conv4_block14_1_bn/AssignNewValue_1"
 �������"  "�  "�Ȱ�7"
	 �줺��"6*2sequential/densenet169/conv4_block14_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�Е��7"
	 ������"6*2sequential/densenet169/conv4_block14_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R���7Ј�"��" *$$1""8�"�  "�����7"
	 ������"6*2sequential/densenet169/conv4_block14_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block14_2_conv/Conv2D"
 �������"  "�  "�����7"
	 ������"6*2sequential/densenet169/conv4_block14_2_conv/Conv2D"
 �������"  "�  "�����7"
	 ������"6*2sequential/densenet169/conv4_block14_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "��Џ�7"
	 ������">*:sequential/densenet169/conv4_block15_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ǫ��">*:sequential/densenet169/conv4_block15_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[704]"�  "�Е��7"
	 ������">*:sequential/densenet169/conv4_block15_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[704]"�  "�����7"
	 �����">*:sequential/densenet169/conv4_block15_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[704]"�  "�����7"
	 �����">*:sequential/densenet169/conv4_block15_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[704]"�  "CU����7���"��" *$$1"!*size:2816 dest:0 async:1"�  "CU�̋�7���"��" *$$1"!*size:2816 dest:0 async:1"�  ",`����7�"��" *$$1""8�"�  ",W����7�ȁ"��" *$$1""8�"�  "�����7"
	 �ꋥ��"<*8sequential/densenet169/conv4_block15_0_bn/AssignNewValue"
 �������"  "�  "�����7"
	 ��ǫ��">*:sequential/densenet169/conv4_block15_0_bn/AssignNewValue_1"
 �������"  "�  "��Џ�7"
	 �����"6*2sequential/densenet169/conv4_block15_1_conv/Conv2D"
 �������"
*output" "*
	 �����"6*2sequential/densenet169/conv4_block15_1_conv/Conv2D"
 �������"*temp" "*
	 �����"6*2sequential/densenet169/conv4_block15_1_conv/Conv2D"
 �������"  "�  "�����7"
	 �����">*:sequential/densenet169/conv4_block15_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �ڋ���">*:sequential/densenet169/conv4_block15_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��˫�7"
	 �ދ���">*:sequential/densenet169/conv4_block15_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����7"
	 ��ƫ��">*:sequential/densenet169/conv4_block15_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����7"
	 ��ƫ��">*:sequential/densenet169/conv4_block15_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BUȇ��7���"��" *$$1"!*size:512 dest:0 async:1"�  "BU��́8���"��" *$$1"!*size:512 dest:0 async:1"�  ",`؅��8�Ǹ"��" *$$1""8�"�  ",W���8�܊"��" *$$1""8�"�  "���Տ8"
	 ������"<*8sequential/densenet169/conv4_block15_1_bn/AssignNewValue"
 �������"  "�  "����8"
	 ������">*:sequential/densenet169/conv4_block15_1_bn/AssignNewValue_1"
 �������"  "�  "����8"
	 �줺��"6*2sequential/densenet169/conv4_block15_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�к��8"
	 �����"6*2sequential/densenet169/conv4_block15_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����8��"��" *$$1""8�"�  "��ɝ�8"
	 ������"6*2sequential/densenet169/conv4_block15_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block15_2_conv/Conv2D"
 �������"  "�  "���֦8"
	 �����"6*2sequential/densenet169/conv4_block15_2_conv/Conv2D"
 �������"  "�  "�����8"
	 �����"6*2sequential/densenet169/conv4_block15_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "��Τ�8"
	 ������">*:sequential/densenet169/conv4_block16_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �ꋥ��">*:sequential/densenet169/conv4_block16_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[736]"�  "�����8"
	 ������">*:sequential/densenet169/conv4_block16_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[736]"�  "���ϳ8"
	 �Ƃ���">*:sequential/densenet169/conv4_block16_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[736]"�  "�����8"
	 ������">*:sequential/densenet169/conv4_block16_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[736]"�  "CU��ǵ8���"��" *$$1"!*size:2944 dest:0 async:1"�  "CU�ู8���"��" *$$1"!*size:2944 dest:0 async:1"�  ",`��Ծ8��"��" *$$1""8�"�  ",W����8���"��" *$$1""8�"�  "��ƣ�8"
	 �Ό���"<*8sequential/densenet169/conv4_block16_0_bn/AssignNewValue"
 �������"  "�  "�����8"
	 ��ȫ��">*:sequential/densenet169/conv4_block16_0_bn/AssignNewValue_1"
 �������"  "�  "�����8"
	 ������"6*2sequential/densenet169/conv4_block16_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block16_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv4_block16_1_conv/Conv2D"
 �������"  "�  "�����8"
	 ������">*:sequential/densenet169/conv4_block16_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ƫ��">*:sequential/densenet169/conv4_block16_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����8"
	 ��ƫ��">*:sequential/densenet169/conv4_block16_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����8"
	 ������">*:sequential/densenet169/conv4_block16_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����8"
	 ������">*:sequential/densenet169/conv4_block16_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BUؓ��8���"��" *$$1"!*size:512 dest:0 async:1"�  "BU�ɾ�8���"��" *$$1"!*size:512 dest:0 async:1"�  ",`��8���"��" *$$1""8�"�  ",Wب��8���"��" *$$1""8�"�  "�����8"
	 ������"<*8sequential/densenet169/conv4_block16_1_bn/AssignNewValue"
 �������"  "�  "�����8"
	 ������">*:sequential/densenet169/conv4_block16_1_bn/AssignNewValue_1"
 �������"  "�  "�����8"
	 �줺��"6*2sequential/densenet169/conv4_block16_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����8"
	 ������"6*2sequential/densenet169/conv4_block16_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����8���"��" *$$1""8�"�  "�����8"
	 ������"6*2sequential/densenet169/conv4_block16_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block16_2_conv/Conv2D"
 �������"  "�  "���΋9"
	 ������"6*2sequential/densenet169/conv4_block16_2_conv/Conv2D"
 �������"  "�  "�����9"
	 ������"6*2sequential/densenet169/conv4_block16_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "���ŗ9"
	 ������">*:sequential/densenet169/conv4_block17_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ȫ��">*:sequential/densenet169/conv4_block17_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[768]"�  "��Ļ�9"
	 ������">*:sequential/densenet169/conv4_block17_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[768]"�  "����9"
	 �Ό���">*:sequential/densenet169/conv4_block17_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[768]"�  "��ݓ�9"
	 �����">*:sequential/densenet169/conv4_block17_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[768]"�  "CU��9���"��" *$$1"!*size:3072 dest:0 async:1"�  "CUȂ��9���"��" *$$1"!*size:3072 dest:0 async:1"�  ",`����9ؿ�"��" *$$1""8�"�  ",W���9���"��" *$$1""8�"�  "���Ȭ9"
	 ������"<*8sequential/densenet169/conv4_block17_0_bn/AssignNewValue"
 �������"  "�  "���˭9"
	 ��ȫ��">*:sequential/densenet169/conv4_block17_0_bn/AssignNewValue_1"
 �������"  "�  "��Ѻ�9"
	 ������"6*2sequential/densenet169/conv4_block17_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block17_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv4_block17_1_conv/Conv2D"
 �������"  "�  "�����9"
	 ������">*:sequential/densenet169/conv4_block17_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv4_block17_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�خ��9"
	 ������">*:sequential/densenet169/conv4_block17_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����9"
	 ��ǫ��">*:sequential/densenet169/conv4_block17_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ޛ�9"
	 ��ǫ��">*:sequential/densenet169/conv4_block17_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����9��"��" *$$1"!*size:512 dest:0 async:1"�  "BU���9���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����9���"��" *$$1""8�"�  ",W�ֈ�9��"��" *$$1""8�"�  "�����9"
	 ������"<*8sequential/densenet169/conv4_block17_1_bn/AssignNewValue"
 �������"  "�  "�����9"
	 ������">*:sequential/densenet169/conv4_block17_1_bn/AssignNewValue_1"
 �������"  "�  "�С��9"
	 �줺��"6*2sequential/densenet169/conv4_block17_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����9"
	 ������"6*2sequential/densenet169/conv4_block17_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����9賕"��" *$$1""8�"�  "�����9"
	 ������"6*2sequential/densenet169/conv4_block17_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block17_2_conv/Conv2D"
 �������"  "�  "��ʉ�9"
	 ������"6*2sequential/densenet169/conv4_block17_2_conv/Conv2D"
 �������"  "�  "�����9"
	 ������"6*2sequential/densenet169/conv4_block17_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "�����9"
	 �̃���">*:sequential/densenet169/conv4_block18_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����">*:sequential/densenet169/conv4_block18_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[800]"�  "��Ν�9"
	 ������">*:sequential/densenet169/conv4_block18_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[800]"�  "�����9"
	 �����">*:sequential/densenet169/conv4_block18_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[800]"�  "�����9"
	 �����">*:sequential/densenet169/conv4_block18_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[800]"�  "CU����9���"��" *$$1"!*size:3200 dest:0 async:1"�  "CU����:؆�"��" *$$1"!*size:3200 dest:0 async:1"�  ",`��ņ:���"��" *$$1""8�"�  ",WؕΌ:�ԅ"��" *$$1""8�"�  "���:"
	 ������"<*8sequential/densenet169/conv4_block18_0_bn/AssignNewValue"
 �������"  "�  "�����:"
	 �����">*:sequential/densenet169/conv4_block18_0_bn/AssignNewValue_1"
 �������"  "�  "�Ж��:"
	 �����"6*2sequential/densenet169/conv4_block18_1_conv/Conv2D"
 �������"
*output" "*
	 �����"6*2sequential/densenet169/conv4_block18_1_conv/Conv2D"
 �������"*temp" "*
	 �����"6*2sequential/densenet169/conv4_block18_1_conv/Conv2D"
 �������"  "�  "�𧘩:"
	 �����">*:sequential/densenet169/conv4_block18_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv4_block18_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��Ϛ�:"
	 ������">*:sequential/densenet169/conv4_block18_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����:"
	 ��ǫ��">*:sequential/densenet169/conv4_block18_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��Ϭ�:"
	 ��ǫ��">*:sequential/densenet169/conv4_block18_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�뷬:���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����:ȯ�"��" *$$1"!*size:512 dest:0 async:1"�  ",`�ѵ:��"��" *$$1""8�"�  ",W��к:���"��" *$$1""8�"�  "�����:"
	 ������"<*8sequential/densenet169/conv4_block18_1_bn/AssignNewValue"
 �������"  "�  "��᣿:"
	 ������">*:sequential/densenet169/conv4_block18_1_bn/AssignNewValue_1"
 �������"  "�  "�،��:"
	 �줺��"6*2sequential/densenet169/conv4_block18_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����:"
	 ������"6*2sequential/densenet169/conv4_block18_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����:���"��" *$$1""8�"�  "�����:"
	 ������"6*2sequential/densenet169/conv4_block18_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block18_2_conv/Conv2D"
 �������"  "�  "�����:"
	 ������"6*2sequential/densenet169/conv4_block18_2_conv/Conv2D"
 �������"  "�  "�����:"
	 ������"6*2sequential/densenet169/conv4_block18_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "�����:"
	 ������">*:sequential/densenet169/conv4_block19_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����">*:sequential/densenet169/conv4_block19_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[832]"�  "�����:"
	 ������">*:sequential/densenet169/conv4_block19_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[832]"�  "����:"
	 ��ۅ��">*:sequential/densenet169/conv4_block19_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[832]"�  "�����:"
	 ��ۅ��">*:sequential/densenet169/conv4_block19_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[832]"�  "CU����:���"��" *$$1"!*size:3328 dest:0 async:1"�  "CU����:���"��" *$$1"!*size:3328 dest:0 async:1"�  ",`�ʳ�:���"��" *$$1""8�"�  ",W���:���"��" *$$1""8�"�  "�����:"
	 ��Ŧ��"<*8sequential/densenet169/conv4_block19_0_bn/AssignNewValue"
 �������"  "�  "����:"
	 �����">*:sequential/densenet169/conv4_block19_0_bn/AssignNewValue_1"
 �������"  "�  "�����:"
	 �����"6*2sequential/densenet169/conv4_block19_1_conv/Conv2D"
 �������"
*output" "*
	 �����"6*2sequential/densenet169/conv4_block19_1_conv/Conv2D"
 �������"*temp" "*
	 �����"6*2sequential/densenet169/conv4_block19_1_conv/Conv2D"
 �������"  "�  "�Ѓ��;"
	 �����">*:sequential/densenet169/conv4_block19_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ǫ��">*:sequential/densenet169/conv4_block19_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��辍;"
	 ������">*:sequential/densenet169/conv4_block19_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����;"
	 ������">*:sequential/densenet169/conv4_block19_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����;"
	 ��ȫ��">*:sequential/densenet169/conv4_block19_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����;���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����;���"��" *$$1"!*size:512 dest:0 async:1"�  ",`��ח;���"��" *$$1""8�"�  ",W��Ԝ;���"��" *$$1""8�"�  "�����;"
	 ������"<*8sequential/densenet169/conv4_block19_1_bn/AssignNewValue"
 �������"  "�  "�����;"
	 ������">*:sequential/densenet169/conv4_block19_1_bn/AssignNewValue_1"
 �������"  "�  "�����;"
	 �줺��"6*2sequential/densenet169/conv4_block19_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "��ԇ�;"
	 ������"6*2sequential/densenet169/conv4_block19_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",RȬ��;���"��" *$$1""8�"�  "��Ĝ�;"
	 ������"6*2sequential/densenet169/conv4_block19_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block19_2_conv/Conv2D"
 �������"  "�  "��Ï�;"
	 ������"6*2sequential/densenet169/conv4_block19_2_conv/Conv2D"
 �������"  "�  "�غ��;"
	 ������"6*2sequential/densenet169/conv4_block19_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "�ػ��;"
	 ������">*:sequential/densenet169/conv4_block20_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��Ŧ��">*:sequential/densenet169/conv4_block20_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[864]"�  "�����;"
	 �Γ���">*:sequential/densenet169/conv4_block20_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[864]"�  "�࿓�;"
	 �꓉��">*:sequential/densenet169/conv4_block20_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[864]"�  "�����;"
	 �����">*:sequential/densenet169/conv4_block20_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[864]"�  "CU����;���"��" *$$1"!*size:3456 dest:0 async:1"�  "CU���;�Ў"��" *$$1"!*size:3456 dest:0 async:1"�  ",`���;���"��" *$$1""8�"�  ",W����;���"��" *$$1""8�"�  "��˜�;"
	 ��Ʀ��"<*8sequential/densenet169/conv4_block20_0_bn/AssignNewValue"
 �������"  "�  "��ĸ�;"
	 ������">*:sequential/densenet169/conv4_block20_0_bn/AssignNewValue_1"
 �������"  "�  "�����;"
	 ������"6*2sequential/densenet169/conv4_block20_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block20_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv4_block20_1_conv/Conv2D"
 �������"  "�  "�����;"
	 ������">*:sequential/densenet169/conv4_block20_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv4_block20_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����;"
	 ������">*:sequential/densenet169/conv4_block20_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����;"
	 ��ȫ��">*:sequential/densenet169/conv4_block20_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����;"
	 ��ȫ��">*:sequential/densenet169/conv4_block20_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�ٷ�;��"��" *$$1"!*size:512 dest:0 async:1"�  "BU����;���"��" *$$1"!*size:512 dest:0 async:1"�  ",`�Ǖ�;��"��" *$$1""8�"�  ",W����<���"��" *$$1""8�"�  "���<"
	 ��Ʀ��"<*8sequential/densenet169/conv4_block20_1_bn/AssignNewValue"
 �������"  "�  "���ӆ<"
	 ��Ʀ��">*:sequential/densenet169/conv4_block20_1_bn/AssignNewValue_1"
 �������"  "�  "���ˈ<"
	 �줺��"6*2sequential/densenet169/conv4_block20_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "���Ӊ<"
	 ������"6*2sequential/densenet169/conv4_block20_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R��͊<���"��" *$$1""8�"�  "����<"
	 ������"6*2sequential/densenet169/conv4_block20_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block20_2_conv/Conv2D"
 �������"  "�  "��̉�<"
	 ������"6*2sequential/densenet169/conv4_block20_2_conv/Conv2D"
 �������"  "�  "�����<"
	 ������"6*2sequential/densenet169/conv4_block20_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "���ɨ<"
	 ������">*:sequential/densenet169/conv4_block21_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv4_block21_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[896]"�  "�����<"
	 �����">*:sequential/densenet169/conv4_block21_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[896]"�  "����<"
	 ��Ʀ��">*:sequential/densenet169/conv4_block21_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[896]"�  "����<"
	 ������">*:sequential/densenet169/conv4_block21_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[896]"�  "CU���<���"��" *$$1"!*size:3584 dest:0 async:1"�  "CU���<���"��" *$$1"!*size:3584 dest:0 async:1"�  ",`����<���"��" *$$1""8�"�  ",W�̕�<���"��" *$$1""8�"�  "�����<"
	 ��Ǧ��"<*8sequential/densenet169/conv4_block21_0_bn/AssignNewValue"
 �������"  "�  "�����<"
	 ������">*:sequential/densenet169/conv4_block21_0_bn/AssignNewValue_1"
 �������"  "�  "����<"
	 ������"6*2sequential/densenet169/conv4_block21_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block21_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv4_block21_1_conv/Conv2D"
 �������"  "�  "�����<"
	 ������">*:sequential/densenet169/conv4_block21_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��Ʀ��">*:sequential/densenet169/conv4_block21_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����<"
	 ��Ʀ��">*:sequential/densenet169/conv4_block21_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����<"
	 ��ȫ��">*:sequential/densenet169/conv4_block21_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��Υ�<"
	 ��ȫ��">*:sequential/densenet169/conv4_block21_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�¦�<���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����<���"��" *$$1"!*size:512 dest:0 async:1"�  ",`ش��<��"��" *$$1""8�"�  ",W����<�Ǆ"��" *$$1""8�"�  "�����<"
	 ��Ǧ��"<*8sequential/densenet169/conv4_block21_1_bn/AssignNewValue"
 �������"  "�  "�ؼ��<"
	 ��Ǧ��">*:sequential/densenet169/conv4_block21_1_bn/AssignNewValue_1"
 �������"  "�  "�����<"
	 �줺��"6*2sequential/densenet169/conv4_block21_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "��ǎ�<"
	 ������"6*2sequential/densenet169/conv4_block21_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R���<�ô"��" *$$1""8�"�  "�����<"
	 ������"6*2sequential/densenet169/conv4_block21_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block21_2_conv/Conv2D"
 �������"  "�  "�����="
	 ������"6*2sequential/densenet169/conv4_block21_2_conv/Conv2D"
 �������"  "�  "��ߎ="
	 ������"6*2sequential/densenet169/conv4_block21_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "��Ӿ�="
	 �̑���">*:sequential/densenet169/conv4_block22_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv4_block22_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[928]"�  "���="
	 ��Ǧ��">*:sequential/densenet169/conv4_block22_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[928]"�  "����="
	 ������">*:sequential/densenet169/conv4_block22_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[928]"�  "�����="
	 ������">*:sequential/densenet169/conv4_block22_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[928]"�  "CU𘬛=��"��" *$$1"!*size:3712 dest:0 async:1"�  "CU��ԟ=���"��" *$$1"!*size:3712 dest:0 async:1"�  ",`𩵤=���"��" *$$1""8�"�  ",W����=���"��" *$$1""8�"�  "����="
	 ��Ȧ��"<*8sequential/densenet169/conv4_block22_0_bn/AssignNewValue"
 �������"  "�  "�����="
	 �����">*:sequential/densenet169/conv4_block22_0_bn/AssignNewValue_1"
 �������"  "�  "�����="
	 ������"6*2sequential/densenet169/conv4_block22_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block22_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv4_block22_1_conv/Conv2D"
 �������"  "�  "��Ӛ�="
	 ������">*:sequential/densenet169/conv4_block22_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ȫ��">*:sequential/densenet169/conv4_block22_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����="
	 ��Ǧ��">*:sequential/densenet169/conv4_block22_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��׃�="
	 ��Ǧ��">*:sequential/densenet169/conv4_block22_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ڱ�="
	 �����">*:sequential/densenet169/conv4_block22_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�׼�=ع�"��" *$$1"!*size:512 dest:0 async:1"�  "BU����=���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����=���"��" *$$1""8�"�  ",W����=�"��" *$$1""8�"�  "��۶�="
	 ��Ȧ��"<*8sequential/densenet169/conv4_block22_1_bn/AssignNewValue"
 �������"  "�  "�����="
	 ��Ȧ��">*:sequential/densenet169/conv4_block22_1_bn/AssignNewValue_1"
 �������"  "�  "�ా�="
	 �줺��"6*2sequential/densenet169/conv4_block22_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����="
	 ������"6*2sequential/densenet169/conv4_block22_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����=���"��" *$$1""8�"�  "�����="
	 �����"6*2sequential/densenet169/conv4_block22_2_conv/Conv2D"
 �������"*temp" "
	 �����"6*2sequential/densenet169/conv4_block22_2_conv/Conv2D"
 �������"  "�  "�����="
	 ������"6*2sequential/densenet169/conv4_block22_2_conv/Conv2D"
 �������"  "�  "�����="
	 ������"6*2sequential/densenet169/conv4_block22_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "�����="
	 ������">*:sequential/densenet169/conv4_block23_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����">*:sequential/densenet169/conv4_block23_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[960]"�  "�����="
	 ��Ȧ��">*:sequential/densenet169/conv4_block23_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[960]"�  "�����="
	 ������">*:sequential/densenet169/conv4_block23_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[960]"�  "�����>"
	 ������">*:sequential/densenet169/conv4_block23_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[960]"�  "CU����>���"��" *$$1"!*size:3840 dest:0 async:1"�  "CU���>���"��" *$$1"!*size:3840 dest:0 async:1"�  ",`��Պ>�ˈ"��" *$$1""8�"�  ",WȘ��>���"��" *$$1""8�"�  "�ض��>"
	 ��ɦ��"<*8sequential/densenet169/conv4_block23_0_bn/AssignNewValue"
 �������"  "�  "�����>"
	 �����">*:sequential/densenet169/conv4_block23_0_bn/AssignNewValue_1"
 �������"  "�  "����>"
	 ������"6*2sequential/densenet169/conv4_block23_1_conv/Conv2D"
 �������"
*output" "*
	 �����"6*2sequential/densenet169/conv4_block23_1_conv/Conv2D"
 �������"*temp" "*
	 �����"6*2sequential/densenet169/conv4_block23_1_conv/Conv2D"
 �������"  "�  "����>"
	 �����">*:sequential/densenet169/conv4_block23_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��Ȧ��">*:sequential/densenet169/conv4_block23_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����>"
	 ��Ȧ��">*:sequential/densenet169/conv4_block23_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���֬>"
	 �����">*:sequential/densenet169/conv4_block23_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����>"
	 �����">*:sequential/densenet169/conv4_block23_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�膮>���"��" *$$1"!*size:512 dest:0 async:1"�  "BU�ڌ�>��"��" *$$1"!*size:512 dest:0 async:1"�  ",`��۶>���"��" *$$1""8�"�  ",W��Ի>��"��" *$$1""8�"�  "�����>"
	 ��ɦ��"<*8sequential/densenet169/conv4_block23_1_bn/AssignNewValue"
 �������"  "�  "���>"
	 ��ɦ��">*:sequential/densenet169/conv4_block23_1_bn/AssignNewValue_1"
 �������"  "�  "��Ѻ�>"
G��?" ����" ��" ��"
	 �줺��"6*2sequential/densenet169/conv4_block23_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "����>"
	 �����"6*2sequential/densenet169/conv4_block23_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����>���"��" *$$1""8�"�  "��Ѣ�>"
	 �����"6*2sequential/densenet169/conv4_block23_2_conv/Conv2D"
 �������"*temp" "
	 �����"6*2sequential/densenet169/conv4_block23_2_conv/Conv2D"
 �������"  "�  "�����>"
G��?" ����" ��	" ��	"
	 �����"6*2sequential/densenet169/conv4_block23_2_conv/Conv2D"
 �������"  "�  "�����>"
	 �����"6*2sequential/densenet169/conv4_block23_concat/concat"
 �������"
*output" "*
	 �줺��"
  "  "�  "�����>"
	 ������">*:sequential/densenet169/conv4_block24_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ɦ��">*:sequential/densenet169/conv4_block24_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[992]"�  "�����>"
	 ������">*:sequential/densenet169/conv4_block24_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[992]"�  "�ت��>"
	 �Қ���">*:sequential/densenet169/conv4_block24_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[992]"�  "����>"
	 ������">*:sequential/densenet169/conv4_block24_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[992]"�  "CUؤ��>���"��" *$$1"!*size:3968 dest:0 async:1"�  "CU����>���"��" *$$1"!*size:3968 dest:0 async:1"�  ",`����>���"��" *$$1""8�"�  ",W����>���"��" *$$1""8�"�  "�����>"
	 ��ʦ��"<*8sequential/densenet169/conv4_block24_0_bn/AssignNewValue"
 �������"  "�  "�����>"
	 �����">*:sequential/densenet169/conv4_block24_0_bn/AssignNewValue_1"
 �������"  "�  "�����>"
	 ������"6*2sequential/densenet169/conv4_block24_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block24_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv4_block24_1_conv/Conv2D"
 �������"  "�  "�����?"
	 ������">*:sequential/densenet169/conv4_block24_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ɦ��">*:sequential/densenet169/conv4_block24_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�о��?"
	 ��ɦ��">*:sequential/densenet169/conv4_block24_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���Ϗ?"
	 �����">*:sequential/densenet169/conv4_block24_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����?"
	 �����">*:sequential/densenet169/conv4_block24_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����?���"��" *$$1"!*size:512 dest:0 async:1"�  "BU���?���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����?�Ȱ"��" *$$1""8�"�  ",W��ƞ?���"��" *$$1""8�"�  "����?"
	 ��˦��"<*8sequential/densenet169/conv4_block24_1_bn/AssignNewValue"
 �������"  "�  "�����?"
	 ��˦��">*:sequential/densenet169/conv4_block24_1_bn/AssignNewValue_1"
 �������"  "�  "�����?"
	 �줺��"6*2sequential/densenet169/conv4_block24_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����?"
	 ������"6*2sequential/densenet169/conv4_block24_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R�ྨ?���"��" *$$1""8�"�  "����?"
	 ������"6*2sequential/densenet169/conv4_block24_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block24_2_conv/Conv2D"
 �������"  "�  "����?"
	 ������"6*2sequential/densenet169/conv4_block24_2_conv/Conv2D"
 �������"  "�  "�����?"
	 ������"6*2sequential/densenet169/conv4_block24_concat/concat"
 �������"
*output" "*[200,1024,2,2]"�  ",N��Ͼ?���"��" *$$1""8�"�  ",Nȓ��?���"��" *$$1""8�"�  "w����?"
	 �줺��"
  "  "�  "�У��?"
	 ������">*:sequential/densenet169/conv4_block25_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1024,2,2]"�  "�����?"
	 �����">*:sequential/densenet169/conv4_block25_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1024]"�  "�����?"
	 �ط���">*:sequential/densenet169/conv4_block25_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1024]"�  "�����?"
	 ��ʦ��">*:sequential/densenet169/conv4_block25_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1024]"�  "�����?"
	 ������">*:sequential/densenet169/conv4_block25_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1024]"�  "CU����?���"��" *$$1"!*size:4096 dest:0 async:1"�  "CUЊ��?���"��" *$$1"!*size:4096 dest:0 async:1"�  ",`�ۈ�?؞�"��" *$$1""8�"�  ",W����?���"��" *$$1""8�"�  "�����?"
	 ��˦��"<*8sequential/densenet169/conv4_block25_0_bn/AssignNewValue"
 �������"  "�  "�����?"
	 �����">*:sequential/densenet169/conv4_block25_0_bn/AssignNewValue_1"
 �������"  "�  "�蛺�?"
	 ������"6*2sequential/densenet169/conv4_block25_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block25_1_conv/Conv2D"
 �������"*temp" "*[128,1024,1,1]"�  ",R����?��"��" *$$1""8�"�  ",a����?���"��" *$$1""8�"�  "�����?"
	 ������"6*2sequential/densenet169/conv4_block25_1_conv/Conv2D"
 �������"  "�  "��ϟ�?"
	 ������">*:sequential/densenet169/conv4_block25_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����">*:sequential/densenet169/conv4_block25_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����?"
	 ��˦��">*:sequential/densenet169/conv4_block25_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����?"
	 ��˦��">*:sequential/densenet169/conv4_block25_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����?"
	 ������">*:sequential/densenet169/conv4_block25_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����?���"��" *$$1"!*size:512 dest:0 async:1"�  "BU�͒�?賕"��" *$$1"!*size:512 dest:0 async:1"�  ",`����?��"��" *$$1""8�"�  ",Wȫ�@��"��" *$$1""8�"�  "�����@"
	 ��̦��"<*8sequential/densenet169/conv4_block25_1_bn/AssignNewValue"
 �������"  "�  "�ȅ��@"
	 ��̦��">*:sequential/densenet169/conv4_block25_1_bn/AssignNewValue_1"
 �������"  "�  "�����@"
	 �줺��"6*2sequential/densenet169/conv4_block25_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����@"
	 ������"6*2sequential/densenet169/conv4_block25_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����@ج�"��" *$$1""8�"�  "�����@"
	 ������"6*2sequential/densenet169/conv4_block25_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block25_2_conv/Conv2D"
 �������"  "�  "����@"
	 ������"6*2sequential/densenet169/conv4_block25_2_conv/Conv2D"
 �������"  "�  "���Р@"
	 ������"6*2sequential/densenet169/conv4_block25_concat/concat"
 �������"
*output" "*[200,1056,2,2]"�  ",N���@���"��" *$$1""8�"�  ",N؟��@Ȅ�"��" *$$1""8�"�  "w����@"
	 �줺��"
  "  "�  "����@"
	 ������">*:sequential/densenet169/conv4_block26_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1056,2,2]"�  "���ª@"
	 �΀���">*:sequential/densenet169/conv4_block26_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1056]"�  "����@"
	 ��˦��">*:sequential/densenet169/conv4_block26_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1056]"�  "��檫@"
	 ������">*:sequential/densenet169/conv4_block26_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1056]"�  "���ҫ@"
	 ������">*:sequential/densenet169/conv4_block26_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1056]"�  "CU��ެ@���"��" *$$1"!*size:4224 dest:0 async:1"�  "CU��߰@���"��" *$$1"!*size:4224 dest:0 async:1"�  ",`����@�ŏ"��" *$$1""8�"�  ",W��μ@л�"��" *$$1""8�"�  "�����@"
	 ��̦��"<*8sequential/densenet169/conv4_block26_0_bn/AssignNewValue"
 �������"  "�  "����@"
	 �����">*:sequential/densenet169/conv4_block26_0_bn/AssignNewValue_1"
 �������"  "�  "�����@"
	 �����"6*2sequential/densenet169/conv4_block26_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block26_1_conv/Conv2D"
 �������"*temp" "*[128,1056,1,1]"�  ",R����@���"��" *$$1""8�"�  ",a����@���"��" *$$1""8�"�  "�����@"
	 ������"6*2sequential/densenet169/conv4_block26_1_conv/Conv2D"
 �������"  "�  "�؍��@"
	 ������">*:sequential/densenet169/conv4_block26_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��̦��">*:sequential/densenet169/conv4_block26_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ʅ�@"
	 ��̦��">*:sequential/densenet169/conv4_block26_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��Գ�@"
	 ������">*:sequential/densenet169/conv4_block26_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�Ȯ��@"
	 ������">*:sequential/densenet169/conv4_block26_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����@���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����@���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����@���"��" *$$1""8�"�  ",Wؾ��@���"��" *$$1""8�"�  "�����@"
	 ��ͦ��"<*8sequential/densenet169/conv4_block26_1_bn/AssignNewValue"
 �������"  "�  "����@"
	 ��ͦ��">*:sequential/densenet169/conv4_block26_1_bn/AssignNewValue_1"
 �������"  "�  "�Ƞ��@"
	 �줺��"6*2sequential/densenet169/conv4_block26_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����@"
	 ������"6*2sequential/densenet169/conv4_block26_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����@�ҝ"��" *$$1""8�"�  "�९�@"
	 ������"6*2sequential/densenet169/conv4_block26_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block26_2_conv/Conv2D"
 �������"  "�  "��ŉ�A"
	 ������"6*2sequential/densenet169/conv4_block26_2_conv/Conv2D"
 �������"  "�  "�����A"
	 ������"6*2sequential/densenet169/conv4_block26_concat/concat"
 �������"
*output" "*[200,1088,2,2]"�  ",N��A���"��" *$$1""8�"�  ",N谔�A���"��" *$$1""8�"�  "w����A"
	 �줺��"
  "  "�  "�����A"
	 ������">*:sequential/densenet169/conv4_block27_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1088,2,2]"�  "���ȕA"
	 �����">*:sequential/densenet169/conv4_block27_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1088]"�  "�����A"
	 ��̦��">*:sequential/densenet169/conv4_block27_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1088]"�  "��۵�A"
	 ������">*:sequential/densenet169/conv4_block27_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1088]"�  "�����A"
	 ������">*:sequential/densenet169/conv4_block27_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1088]"�  "CU��ʘA���"��" *$$1"!*size:4352 dest:0 async:1"�  "CU��ߜAн�"��" *$$1"!*size:4352 dest:0 async:1"�  ",`����A���"��" *$$1""8�"�  ",W����A���"��" *$$1""8�"�  "����A"
	 ��ͦ��"<*8sequential/densenet169/conv4_block27_0_bn/AssignNewValue"
 �������"  "�  "��ގ�A"
	 ������">*:sequential/densenet169/conv4_block27_0_bn/AssignNewValue_1"
 �������"  "�  "�𮕰A"
	 �����"6*2sequential/densenet169/conv4_block27_1_conv/Conv2D"
 �������"
*output" "*
	 �����"6*2sequential/densenet169/conv4_block27_1_conv/Conv2D"
 �������"*temp" "*[128,1088,1,1]"�  ",R���A�ڪ"��" *$$1""8�"�  ",a����A���"��" *$$1""8�"�  "����A"
	 �����"6*2sequential/densenet169/conv4_block27_1_conv/Conv2D"
 �������"  "�  "�����A"
	 �����">*:sequential/densenet169/conv4_block27_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ͦ��">*:sequential/densenet169/conv4_block27_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ã�A"
	 ��ͦ��">*:sequential/densenet169/conv4_block27_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����A"
	 ������">*:sequential/densenet169/conv4_block27_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����A"
	 ������">*:sequential/densenet169/conv4_block27_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU���A���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����A���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����Aت�"��" *$$1""8�"�  ",W���A���"��" *$$1""8�"�  "�����A"
	 �����"<*8sequential/densenet169/conv4_block27_1_bn/AssignNewValue"
 �������"  "�  "�����A"
	 �����">*:sequential/densenet169/conv4_block27_1_bn/AssignNewValue_1"
 �������"  "�  "�����A"
	 �줺��"6*2sequential/densenet169/conv4_block27_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "����A"
	 ������"6*2sequential/densenet169/conv4_block27_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����A���"��" *$$1""8�"�  "�����A"
	 ������"6*2sequential/densenet169/conv4_block27_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block27_2_conv/Conv2D"
 �������"  "�  "�����A"
	 ������"6*2sequential/densenet169/conv4_block27_2_conv/Conv2D"
 �������"  "�  "�����A"
	 ������"6*2sequential/densenet169/conv4_block27_concat/concat"
 �������"
*output" "*[200,1120,2,2]"�  ",N����A���"��" *$$1""8�"�  ",N�ט�A�ξ"��" *$$1""8�"�  "w���A"
	 �줺��"
  "  "�  "����A"
	 �̝���">*:sequential/densenet169/conv4_block28_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1120,2,2]"�  "�����A"
	 ��ͦ��">*:sequential/densenet169/conv4_block28_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1120]"�  "��Ϋ�A"
	 ������">*:sequential/densenet169/conv4_block28_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1120]"�  "�Ȯ��A"
	 ������">*:sequential/densenet169/conv4_block28_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1120]"�  "�м��A"
	 ������">*:sequential/densenet169/conv4_block28_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1120]"�  "CU؛��A���"��" *$$1"!*size:4480 dest:0 async:1"�  "CU����B���"��" *$$1"!*size:4480 dest:0 async:1"�  ",`����B���"��" *$$1""8�"�  ",W��ьB���"��" *$$1""8�"�  "�����B"
	 �����"<*8sequential/densenet169/conv4_block28_0_bn/AssignNewValue"
 �������"  "�  "�����B"
	 ������">*:sequential/densenet169/conv4_block28_0_bn/AssignNewValue_1"
 �������"  "�  "���B"
	 ������"6*2sequential/densenet169/conv4_block28_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block28_1_conv/Conv2D"
 �������"*temp" "*[128,1120,1,1]"�  ",R�Т�B���"��" *$$1""8�"�  ",a����B���"��" *$$1""8�"�  "��콥B"
	 ������"6*2sequential/densenet169/conv4_block28_1_conv/Conv2D"
 �������"  "�  "��ݧB"
	 ������">*:sequential/densenet169/conv4_block28_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv4_block28_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���ߨB"
	 �����">*:sequential/densenet169/conv4_block28_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��슩B"
	 �����">*:sequential/densenet169/conv4_block28_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�Ȕ��B"
	 �����">*:sequential/densenet169/conv4_block28_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU��ŪB���"��" *$$1"!*size:512 dest:0 async:1"�  "BU��̮B�߉"��" *$$1"!*size:512 dest:0 async:1"�  ",`����B���	"��" *$$1""8�"�  ",W����B���"��" *$$1""8�"�  "�Ѐ��B"
	 �����"<*8sequential/densenet169/conv4_block28_1_bn/AssignNewValue"
 �������"  "�  "�Ћ��B"
	 �����">*:sequential/densenet169/conv4_block28_1_bn/AssignNewValue_1"
 �������"  "�  "�����B"
	 �줺��"6*2sequential/densenet169/conv4_block28_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "��߮�B"
	 ������"6*2sequential/densenet169/conv4_block28_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����BУ�"��" *$$1""8�"�  "�����B"
	 ������"6*2sequential/densenet169/conv4_block28_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block28_2_conv/Conv2D"
 �������"  "�  "�ପ�B"
	 ������"6*2sequential/densenet169/conv4_block28_2_conv/Conv2D"
 �������"  "�  "�艐�B"
	 ������"6*2sequential/densenet169/conv4_block28_concat/concat"
 �������"
*output" "*[200,1152,2,2]"�  ",N����B��"��" *$$1""8�"�  ",N����B���"��" *$$1""8�"�  "w覞�B"
	 �줺��"
  "  "�  "�����B"
	 ������">*:sequential/densenet169/conv4_block29_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1152,2,2]"�  "��Ć�B"
	 ������">*:sequential/densenet169/conv4_block29_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1152]"�  "��ܲ�B"
	 �����">*:sequential/densenet169/conv4_block29_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1152]"�  "�����B"
	 �ֹ���">*:sequential/densenet169/conv4_block29_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1152]"�  "�ع��B"
	 ������">*:sequential/densenet169/conv4_block29_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1152]"�  "CU����B���"��" *$$1"!*size:4608 dest:0 async:1"�  "CU����B���"��" *$$1"!*size:4608 dest:0 async:1"�  ",`�̦�B���"��" *$$1""8�"�  ",W����C���"��" *$$1""8�"�  "��ۈ�C"
	 �����"<*8sequential/densenet169/conv4_block29_0_bn/AssignNewValue"
 �������"  "�  "��ª�C"
	 �����">*:sequential/densenet169/conv4_block29_0_bn/AssignNewValue_1"
 �������"  "�  "�觝�C"
	 �����"6*2sequential/densenet169/conv4_block29_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block29_1_conv/Conv2D"
 �������"*temp" "*[128,1152,1,1]"�  ",R�ݟ�C���"��" *$$1""8�"�  ",a�Л�C���"��" *$$1""8�"�  "�����C"
	 ������"6*2sequential/densenet169/conv4_block29_1_conv/Conv2D"
 �������"  "�  "�ؒњC"
	 ������">*:sequential/densenet169/conv4_block29_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����">*:sequential/densenet169/conv4_block29_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���ޛC"
	 �����">*:sequential/densenet169/conv4_block29_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�Ȋ��C"
	 �����">*:sequential/densenet169/conv4_block29_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��岜C"
	 �����">*:sequential/densenet169/conv4_block29_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����C���"��" *$$1"!*size:512 dest:0 async:1"�  "BU�ڹ�C��"��" *$$1"!*size:512 dest:0 async:1"�  ",`���C��"��" *$$1""8�"�  ",W����C���"��" *$$1""8�"�  "�����C"
	 �����"<*8sequential/densenet169/conv4_block29_1_bn/AssignNewValue"
 �������"  "�  "���ְC"
	 �����">*:sequential/densenet169/conv4_block29_1_bn/AssignNewValue_1"
 �������"  "�  "�АвC"
	 �줺��"6*2sequential/densenet169/conv4_block29_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "����C"
	 ������"6*2sequential/densenet169/conv4_block29_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R��ܴC���"��" *$$1""8�"�  "����C"
fL��?" ����" ���" ���"
	 ������"6*2sequential/densenet169/conv4_block29_2_conv/Conv2D"
 �������"*temp" "
	 ������"6*2sequential/densenet169/conv4_block29_2_conv/Conv2D"
 �������"  "�  "��ϙ�C"
	 ������"6*2sequential/densenet169/conv4_block29_2_conv/Conv2D"
 �������"  "�  "�����C"
	 ������"6*2sequential/densenet169/conv4_block29_concat/concat"
 �������"
*output" "*[200,1184,2,2]"�  ",N����C��"��" *$$1""8�"�  ",N����C���"��" *$$1""8�"�  "w�æ�C"
	 �줺��"
  "  "�  "�ش��C"
	 �̅���">*:sequential/densenet169/conv4_block30_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1184,2,2]"�  "�����C"
	 �����">*:sequential/densenet169/conv4_block30_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1184]"�  "�����C"
	 ������">*:sequential/densenet169/conv4_block30_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1184]"�  "�����C"
	 ������">*:sequential/densenet169/conv4_block30_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1184]"�  "�����C"
	 ������">*:sequential/densenet169/conv4_block30_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1184]"�  "CU����C��"��" *$$1"!*size:4736 dest:0 async:1"�  "CU����C���"��" *$$1"!*size:4736 dest:0 async:1"�  ",`�ղ�C�"��" *$$1""8�"�  ",W�ֲ�C���"��" *$$1""8�"�  "����C"
	 �����"<*8sequential/densenet169/conv4_block30_0_bn/AssignNewValue"
 �������"  "�  "����C"
	 ������">*:sequential/densenet169/conv4_block30_0_bn/AssignNewValue_1"
 �������"  "�  "�����C"
	 �����"6*2sequential/densenet169/conv4_block30_1_conv/Conv2D"
 �������"
*output" "*
�F��?" ����" ��%" ��%"
	 �셇��"6*2sequential/densenet169/conv4_block30_1_conv/Conv2D"
 �������"*temp" "*[128,1184,1,1]"�  ",R����C���"��" *$$1""8�"�  ",a����C���"��" *$$1""8�"�  "�����C"
	 �셇��"6*2sequential/densenet169/conv4_block30_1_conv/Conv2D"
 �������"  "�  "�����C"
	 �셇��">*:sequential/densenet169/conv4_block30_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����">*:sequential/densenet169/conv4_block30_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��˵�D"
	 �����">*:sequential/densenet169/conv4_block30_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����D"
	 �����">*:sequential/densenet169/conv4_block30_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��懁D"
	 �����">*:sequential/densenet169/conv4_block30_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����D���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����D���"��" *$$1"!*size:512 dest:0 async:1"�  ",`���D���"��" *$$1""8�"�  ",W��ۏD��"��" *$$1""8�"�  "�����D"
	 ������"<*8sequential/densenet169/conv4_block30_1_bn/AssignNewValue"
 �������"  "�  "�Ȑ��D"
	 ������">*:sequential/densenet169/conv4_block30_1_bn/AssignNewValue_1"
 �������"  "�  "��ߋ�D"
	 �줺��"6*2sequential/densenet169/conv4_block30_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "���ƘD"
	 �잇��"6*2sequential/densenet169/conv4_block30_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����D�ݶ"��" *$$1""8�"�  "��ו�D"
	 �짇��"6*2sequential/densenet169/conv4_block30_2_conv/Conv2D"
 �������"*temp" "
	 �짇��"6*2sequential/densenet169/conv4_block30_2_conv/Conv2D"
 �������"  "�  "���ΫD"
	 �잇��"6*2sequential/densenet169/conv4_block30_2_conv/Conv2D"
 �������"  "�  "�����D"
	 �잇��"6*2sequential/densenet169/conv4_block30_concat/concat"
 �������"
*output" "*[200,1216,2,2]"�  ",N��ůD���"��" *$$1""8�"�  ",N����D���"��" *$$1""8�"�  "w�Ȯ�D"
	 �줺��"
  "  "�  "����D"
F%u�?" ����" ���" ���"
	 ������">*:sequential/densenet169/conv4_block31_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1216,2,2]"�  "�����D"
	 ������">*:sequential/densenet169/conv4_block31_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1216]"�  "��껹D"
	 �����">*:sequential/densenet169/conv4_block31_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1216]"�  "�����D"
	 ��ë��">*:sequential/densenet169/conv4_block31_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1216]"�  "�����D"
	 ��ë��">*:sequential/densenet169/conv4_block31_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1216]"�  "CU𸩻D���"��" *$$1"!*size:4864 dest:0 async:1"�  "CU𦣿D�"��" *$$1"!*size:4864 dest:0 async:1"�  ",`���Dಲ"��" *$$1""8�"�  ",W؀��D��"��" *$$1""8�"�  "�����D"
	 ������"<*8sequential/densenet169/conv4_block31_0_bn/AssignNewValue"
 �������"  "�  "�����D"
	 ������">*:sequential/densenet169/conv4_block31_0_bn/AssignNewValue_1"
 �������"  "�  "�����D"
	 ������"6*2sequential/densenet169/conv4_block31_1_conv/Conv2D"
 �������"
*output" "*
	 �쒋��"6*2sequential/densenet169/conv4_block31_1_conv/Conv2D"
 �������"*temp" "*[128,1216,1,1]"�  ",R���D���"��" *$$1""8�"�  ",a�ю�Dآ�"��" *$$1""8�"�  "�����D"
	 �쒋��"6*2sequential/densenet169/conv4_block31_1_conv/Conv2D"
 �������"  "�  "�����D"
	 �쒋��">*:sequential/densenet169/conv4_block31_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv4_block31_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����D"
	 ������">*:sequential/densenet169/conv4_block31_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����D"
	 �����">*:sequential/densenet169/conv4_block31_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����D"
	 �����">*:sequential/densenet169/conv4_block31_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU���D���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����D���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����D���"��" *$$1""8�"�  ",W����D��"��" *$$1""8�"�  "�����D"
	 �����"<*8sequential/densenet169/conv4_block31_1_bn/AssignNewValue"
 �������"  "�  "�����D"
	 �����">*:sequential/densenet169/conv4_block31_1_bn/AssignNewValue_1"
 �������"  "�  "�����D"
	 �줺��"6*2sequential/densenet169/conv4_block31_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�讀�D"
	 �쫋��"6*2sequential/densenet169/conv4_block31_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����D���"��" *$$1""8�"�  "�����E"
	 �촋��"6*2sequential/densenet169/conv4_block31_2_conv/Conv2D"
 �������"*temp" "
	 �촋��"6*2sequential/densenet169/conv4_block31_2_conv/Conv2D"
 �������"  "�  "��ʄ�E"
	 �쫋��"6*2sequential/densenet169/conv4_block31_2_conv/Conv2D"
 �������"  "�  "����E"
	 �쫋��"6*2sequential/densenet169/conv4_block31_concat/concat"
 �������"
*output" "*[200,1248,2,2]"�  ",N����E�"��" *$$1""8�"�  ",N���E���"��" *$$1""8�"�  "w��E"
	 �줺��"
  "  "�  "���ܛE"
	 �̟���">*:sequential/densenet169/conv4_block32_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1248,2,2]"�  "���E"
	 ������">*:sequential/densenet169/conv4_block32_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1248]"�  "���ڜE"
��?" ����" �'" �F"
	 ��ë��">*:sequential/densenet169/conv4_block32_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1248]"�  "���E"
	 ������">*:sequential/densenet169/conv4_block32_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1248]"�  "�����E"
	 ������">*:sequential/densenet169/conv4_block32_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1248]"�  "CU��ӞE���"��" *$$1"!*size:4992 dest:0 async:1"�  "CU��ϢE近"��" *$$1"!*size:4992 dest:0 async:1"�  ",`����E���"��" *$$1""8�"�  ",W����E��"��" *$$1""8�"�  "���̰E"
	 �����"<*8sequential/densenet169/conv4_block32_0_bn/AssignNewValue"
 �������"  "�  "�����E"
��?" ����" �'" �("
	 ������">*:sequential/densenet169/conv4_block32_0_bn/AssignNewValue_1"
 �������"  "�  "�н��E"
	 ������"6*2sequential/densenet169/conv4_block32_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv4_block32_1_conv/Conv2D"
 �������"*temp" "*[128,1248,1,1]"�  ",R�E���"��" *$$1""8�"�  ",a�ҽ�E�ӡ"��" *$$1""8�"�  "�����E"
	 ������"6*2sequential/densenet169/conv4_block32_1_conv/Conv2D"
 �������"  "�  "�����E"
	 ������">*:sequential/densenet169/conv4_block32_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����">*:sequential/densenet169/conv4_block32_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����E"
	 �����">*:sequential/densenet169/conv4_block32_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�Ф��E"
	 �����">*:sequential/densenet169/conv4_block32_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����E"
	 �����">*:sequential/densenet169/conv4_block32_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����E���"��" *$$1"!*size:512 dest:0 async:1"�  "BU���E���"��" *$$1"!*size:512 dest:0 async:1"�  ",`Ȭ��E���"��" *$$1""8�"�  ",W����E���"��" *$$1""8�"�  "�ȕ��E"
	 �����"<*8sequential/densenet169/conv4_block32_1_bn/AssignNewValue"
 �������"  "�  "�����E"
	 �����">*:sequential/densenet169/conv4_block32_1_bn/AssignNewValue_1"
 �������"  "�  "����E"
	 �줺��"6*2sequential/densenet169/conv4_block32_2_conv/Conv2D"
 �������"
*output" "*[200,32,2,2]"�  "�����E"
���?" ����" ��	" ��	"
	 ��ŏ��"6*2sequential/densenet169/conv4_block32_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",RЈ��E�Δ"��" *$$1""8�"�  "�����E"
	 ��Ώ��"6*2sequential/densenet169/conv4_block32_2_conv/Conv2D"
 �������"*temp" "
���?" ����" ���" ���"
	 ��Ώ��"6*2sequential/densenet169/conv4_block32_2_conv/Conv2D"
 �������"  "�  "�����E"
	 ��ŏ��"6*2sequential/densenet169/conv4_block32_2_conv/Conv2D"
 �������"  "�  "�����E"
	 ��ŏ��"6*2sequential/densenet169/conv4_block32_concat/concat"
 �������"
*output" "*[200,1280,2,2]"�  ",N����E��"��" *$$1""8�"�  ",N؟��Eؽ�"��" *$$1""8�"�  "wع��E"
	 �줺��"
  "  "�  "�����E"
	 ������"4*0sequential/densenet169/pool4_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1280,2,2]"�  "�ఊ�F"
	 ������"4*0sequential/densenet169/pool4_bn/FusedBatchNormV3"
 �������"
*output" "
*[1280]"�  "�����F"
	 �����"4*0sequential/densenet169/pool4_bn/FusedBatchNormV3"
 �������"
*output" "
*[1280]"�  "����F"
	 ������"4*0sequential/densenet169/pool4_bn/FusedBatchNormV3"
 �������"
*output" "
*[1280]"�  "�����F"
	 ������"4*0sequential/densenet169/pool4_bn/FusedBatchNormV3"
 �������"
*output" "
*[1280]"�  "CU����F��"��" *$$1"!*size:5120 dest:0 async:1"�  "CU����F�ڃ"��" *$$1"!*size:5120 dest:0 async:1"�  ",`��ъF���"��" *$$1""8�"�  ",W�ա�Fఉ"��" *$$1""8�"�  "�����F"
	 �����"2*.sequential/densenet169/pool4_bn/AssignNewValue"
 �������"  "�  "�����F"
	 ������"4*0sequential/densenet169/pool4_bn/AssignNewValue_1"
 �������"  "�  "�����F"
	 ������",*(sequential/densenet169/pool4_conv/Conv2D"
 �������"
*output" "*
	 ������",*(sequential/densenet169/pool4_conv/Conv2D"
 �������"*temp" "*[640,1280,1,1]"�  ",R����F�Ȏ"��" *$$1""8�"�  "���ݠF"
	 �����",*(sequential/densenet169/pool4_conv/Conv2D"
 �������"*temp" "*[32]"�  ",S�֦�F���"��" *$$1""8�"�  ",T���F���"��" *$$1""8�"�  "���תF"
	 �����",*(sequential/densenet169/pool4_conv/Conv2D"
 �������"  "�  "�����F"
	 ������",*(sequential/densenet169/pool4_conv/Conv2D"
 �������"  "�  "�����F"
	 ������"-*)sequential/densenet169/pool4_pool/AvgPool"
 �������"
*output" "*
	 ������"
  "  "�  "��爺F"
	 ������"=*9sequential/densenet169/conv5_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv5_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[640]"�  "���F"
	 ������"=*9sequential/densenet169/conv5_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[640]"�  "�����F"
	 ������"=*9sequential/densenet169/conv5_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[640]"�  "���ֻF"
	 ������"=*9sequential/densenet169/conv5_block1_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[640]"�  "CU���F���"��" *$$1"!*size:2560 dest:0 async:1"�  "CU����F���"��" *$$1"!*size:2560 dest:0 async:1"�  ",`����F���"��" *$$1""8�"�  ",W�ѵ�F���"��" *$$1""8�"�  "��Ƌ�F"
	 �����";*7sequential/densenet169/conv5_block1_0_bn/AssignNewValue"
 �������"  "�  "����F"
	 �����"=*9sequential/densenet169/conv5_block1_0_bn/AssignNewValue_1"
 �������"  "�  "�����F"
	 �줺��"5*1sequential/densenet169/conv5_block1_1_conv/Conv2D"
 �������"
*output" "*
	 ��ؓ��"5*1sequential/densenet169/conv5_block1_1_conv/Conv2D"
 �������"*temp" "*
	 ��ؓ��"5*1sequential/densenet169/conv5_block1_1_conv/Conv2D"
 �������"  "�  "�����F"
	 ��ؓ��"=*9sequential/densenet169/conv5_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv5_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����F"
	 �����"=*9sequential/densenet169/conv5_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����F"
	 ������"=*9sequential/densenet169/conv5_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����F"
	 ������"=*9sequential/densenet169/conv5_block1_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�ڼ�F���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����F���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����F���"��" *$$1""8�"�  ",W����F���"��" *$$1""8�"�  "�ؒ��F"
	 �����";*7sequential/densenet169/conv5_block1_1_bn/AssignNewValue"
 �������"  "�  "�����F"
	 �����"=*9sequential/densenet169/conv5_block1_1_bn/AssignNewValue_1"
 �������"  "�  "�����F"
	 ������"5*1sequential/densenet169/conv5_block1_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�Ș��F"
	 ��ޓ��"5*1sequential/densenet169/conv5_block1_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����F菡"��" *$$1""8�"�  ",a����G���"��" *$$1""8�"�  "�ؒڋG"
	 ��ޓ��"5*1sequential/densenet169/conv5_block1_2_conv/Conv2D"
 �������"  "�  "����G"
	 ��ޓ��"5*1sequential/densenet169/conv5_block1_concat/concat"
 �������"
*output" "*
	 ������"
  "  "�  "����G"
	 ������"=*9sequential/densenet169/conv5_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv5_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[672]"�  "��ݱ�G"
	 ������"=*9sequential/densenet169/conv5_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[672]"�  "�ؔٙG"
	 ������"=*9sequential/densenet169/conv5_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[672]"�  "��灚G"
	 �����"=*9sequential/densenet169/conv5_block2_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[672]"�  "CU����G���"��" *$$1"!*size:2688 dest:0 async:1"�  "CU����G���"��" *$$1"!*size:2688 dest:0 async:1"�  ",`���G���"��" *$$1""8�"�  ",W��ɩG���"��" *$$1""8�"�  "�Ў��G"
	 �����";*7sequential/densenet169/conv5_block2_0_bn/AssignNewValue"
 �������"  "�  "�����G"
	 ��ׄ��"=*9sequential/densenet169/conv5_block2_0_bn/AssignNewValue_1"
 �������"  "�  "���ٱG"
	 ��Ք��"5*1sequential/densenet169/conv5_block2_1_conv/Conv2D"
 �������"
*output" "*
	 ��۔��"5*1sequential/densenet169/conv5_block2_1_conv/Conv2D"
 �������"*temp" "*
	 ��۔��"5*1sequential/densenet169/conv5_block2_1_conv/Conv2D"
 �������"  "�  "�ȡ��G"
	 ��۔��"=*9sequential/densenet169/conv5_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv5_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����G"
	 �����"=*9sequential/densenet169/conv5_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����G"
	 ������"=*9sequential/densenet169/conv5_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����G"
	 ������"=*9sequential/densenet169/conv5_block2_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����G���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����G��"��" *$$1"!*size:512 dest:0 async:1"�  ",`����G���"��" *$$1""8�"�  ",W����G���"��" *$$1""8�"�  "��ٱ�G"
	 �����";*7sequential/densenet169/conv5_block2_1_bn/AssignNewValue"
 �������"  "�  "�����G"
	 �����"=*9sequential/densenet169/conv5_block2_1_bn/AssignNewValue_1"
 �������"  "�  "�����G"
	 ������"5*1sequential/densenet169/conv5_block2_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����G"
	 �����"5*1sequential/densenet169/conv5_block2_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����GЋ�"��" *$$1""8�"�  ",aН��G���"��" *$$1""8�"�  "�����G"
	 �����"5*1sequential/densenet169/conv5_block2_2_conv/Conv2D"
 �������"  "�  "�Љ��G"
	 �����"5*1sequential/densenet169/conv5_block2_concat/concat"
 �������"
*output" "*
	 ������"
  "  "�  "�����G"
	 ������"=*9sequential/densenet169/conv5_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ׄ��"=*9sequential/densenet169/conv5_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[704]"�  "���G"
	 �����"=*9sequential/densenet169/conv5_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[704]"�  "����G"
	 �����"=*9sequential/densenet169/conv5_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[704]"�  "�����G"
	 �����"=*9sequential/densenet169/conv5_block3_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[704]"�  "CU���Gȕ�"��" *$$1"!*size:2816 dest:0 async:1"�  "CU����G�"��" *$$1"!*size:2816 dest:0 async:1"�  ",`��ǃH���"��" *$$1""8�"�  ",W𦤉HȲ�"��" *$$1""8�"�  "�����H"
	 �����";*7sequential/densenet169/conv5_block3_0_bn/AssignNewValue"
 �������"  "�  "�����H"
	 ��؄��"=*9sequential/densenet169/conv5_block3_0_bn/AssignNewValue_1"
 �������"  "�  "�����H"
	 �즕��"5*1sequential/densenet169/conv5_block3_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv5_block3_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv5_block3_1_conv/Conv2D"
 �������"  "�  "�����H"
�R��?" ����" ��" ��"
	 ������"=*9sequential/densenet169/conv5_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv5_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�Ⱦ��H"
	 �����"=*9sequential/densenet169/conv5_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����H"
	 �����"=*9sequential/densenet169/conv5_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���ޣH"
	 ��؄��"=*9sequential/densenet169/conv5_block3_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU��åH���"��" *$$1"!*size:512 dest:0 async:1"�  "BU��өH���"��" *$$1"!*size:512 dest:0 async:1"�  ",`�ޕ�Hؕ�"��" *$$1""8�"�  ",W�ꏳH���"��" *$$1""8�"�  "���ζH"
	 �����";*7sequential/densenet169/conv5_block3_1_bn/AssignNewValue"
 �������"  "�  "���޷H"
	 �����"=*9sequential/densenet169/conv5_block3_1_bn/AssignNewValue_1"
 �������"  "�  "���عH"
	 ������"5*1sequential/densenet169/conv5_block3_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "��ë�H"
	 ������"5*1sequential/densenet169/conv5_block3_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����H菡"��" *$$1""8�"�  ",a����H���"��" *$$1""8�"�  "�����H"
	 ������"5*1sequential/densenet169/conv5_block3_2_conv/Conv2D"
 �������"  "�  "����H"
	 ������"5*1sequential/densenet169/conv5_block3_concat/concat"
 �������"
*output" "*
	 ������"
  "  "�  "�����H"
	 ��ו��"=*9sequential/densenet169/conv5_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv5_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[736]"�  "�����H"
	 ������"=*9sequential/densenet169/conv5_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[736]"�  "�����H"
	 ������"=*9sequential/densenet169/conv5_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[736]"�  "����H"
	 ������"=*9sequential/densenet169/conv5_block4_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[736]"�  "CU����H��"��" *$$1"!*size:2944 dest:0 async:1"�  "CU�Ǖ�H���"��" *$$1"!*size:2944 dest:0 async:1"�  ",`����H���"��" *$$1""8�"�  ",W����I�·"��" *$$1""8�"�  "�����I"
	 �����";*7sequential/densenet169/conv5_block4_0_bn/AssignNewValue"
 �������"  "�  "���I"
	 ��؄��"=*9sequential/densenet169/conv5_block4_0_bn/AssignNewValue_1"
 �������"  "�  "�����I"
	 ������"5*1sequential/densenet169/conv5_block4_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv5_block4_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv5_block4_1_conv/Conv2D"
 �������"  "�  "��ӠI"
	 ������"=*9sequential/densenet169/conv5_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv5_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���СI"
	 �����"=*9sequential/densenet169/conv5_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����I"
	 ��؄��"=*9sequential/densenet169/conv5_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���I"
	 ��؄��"=*9sequential/densenet169/conv5_block4_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU��£I���"��" *$$1"!*size:512 dest:0 async:1"�  "BU��ԧI���"��" *$$1"!*size:512 dest:0 async:1"�  ",`��ެIȵ�"��" *$$1""8�"�  ",W���I���"��" *$$1""8�"�  "�����I"
	 �����";*7sequential/densenet169/conv5_block4_1_bn/AssignNewValue"
 �������"  "�  "���նI"
	 �����"=*9sequential/densenet169/conv5_block4_1_bn/AssignNewValue_1"
 �������"  "�  "���߸I"
	 �܇���"5*1sequential/densenet169/conv5_block4_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����I"
	 ������"5*1sequential/densenet169/conv5_block4_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R���I���"��" *$$1""8�"�  ",a�ĩ�I���"��" *$$1""8�"�  "�����I"
	 ������"5*1sequential/densenet169/conv5_block4_2_conv/Conv2D"
 �������"  "�  "�����I"
	 ������"5*1sequential/densenet169/conv5_block4_concat/concat"
 �������"
*output" "*
	 �܇���"
  "  "�  "�����I"
	 �䮖��"=*9sequential/densenet169/conv5_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��؄��"=*9sequential/densenet169/conv5_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[768]"�  "�����I"
	 �����"=*9sequential/densenet169/conv5_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[768]"�  "�Ȕ��I"
	 ������"=*9sequential/densenet169/conv5_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[768]"�  "��Ǟ�I"
	 ������"=*9sequential/densenet169/conv5_block5_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[768]"�  "CU�ƫ�I���"��" *$$1"!*size:3072 dest:0 async:1"�  "CUȀ��I�"��" *$$1"!*size:3072 dest:0 async:1"�  ",`����I�Ё"��" *$$1""8�"�  ",W����I�"��" *$$1""8�"�  "����I"
	 �����";*7sequential/densenet169/conv5_block5_0_bn/AssignNewValue"
 �������"  "�  "��˔�I"
	 ������"=*9sequential/densenet169/conv5_block5_0_bn/AssignNewValue_1"
 �������"  "�  "����I"
	 ��Ԗ��"5*1sequential/densenet169/conv5_block5_1_conv/Conv2D"
 �������"
*output" "*
	 ��ږ��"5*1sequential/densenet169/conv5_block5_1_conv/Conv2D"
 �������"*temp" "*
	 ��ږ��"5*1sequential/densenet169/conv5_block5_1_conv/Conv2D"
 �������"  "�  "�Ȕ�J"
	 ��ږ��"=*9sequential/densenet169/conv5_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv5_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����J"
	 �����"=*9sequential/densenet169/conv5_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�蝝�J"
	 ��؄��"=*9sequential/densenet169/conv5_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���ĂJ"
	 ��؄��"=*9sequential/densenet169/conv5_block5_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU��ŃJ���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����J�ڈ"��" *$$1"!*size:512 dest:0 async:1"�  ",`��ՌJ���"��" *$$1""8�"�  ",W��ђJ��"��" *$$1""8�"�  "�����J"
	 �����";*7sequential/densenet169/conv5_block5_1_bn/AssignNewValue"
 �������"  "�  "���×J"
	 �����"=*9sequential/densenet169/conv5_block5_1_bn/AssignNewValue_1"
 �������"  "�  "�����J"
	 �܇���"5*1sequential/densenet169/conv5_block5_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "���ӚJ"
	 ������"5*1sequential/densenet169/conv5_block5_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R��ћJ���"��" *$$1""8�"�  ",a����J���"��" *$$1""8�"�  "���ƨJ"
	 ������"5*1sequential/densenet169/conv5_block5_2_conv/Conv2D"
 �������"  "�  "�ЃԫJ"
	 ������"5*1sequential/densenet169/conv5_block5_concat/concat"
 �������"
*output" "*
	 �܇���"
  "  "�  "��Ơ�J"
	 �쇗��"=*9sequential/densenet169/conv5_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv5_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[800]"�  "�����J"
	 ������"=*9sequential/densenet169/conv5_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[800]"�  "�����J"
	 ������"=*9sequential/densenet169/conv5_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[800]"�  "����J"
	 ������"=*9sequential/densenet169/conv5_block6_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[800]"�  "CU���J���"��" *$$1"!*size:3200 dest:0 async:1"�  "CUȥ�J蹓"��" *$$1"!*size:3200 dest:0 async:1"�  ",`����JГ�"��" *$$1""8�"�  ",W����J��"��" *$$1""8�"�  "�����J"
	 �����";*7sequential/densenet169/conv5_block6_0_bn/AssignNewValue"
 �������"  "�  "�����J"
	 ��ۅ��"=*9sequential/densenet169/conv5_block6_0_bn/AssignNewValue_1"
 �������"  "�  "����J"
	 ������"5*1sequential/densenet169/conv5_block6_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv5_block6_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv5_block6_1_conv/Conv2D"
 �������"  "�  "����J"
	 ������"=*9sequential/densenet169/conv5_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv5_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ԟ�J"
	 �����"=*9sequential/densenet169/conv5_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����J"
	 ��ۅ��"=*9sequential/densenet169/conv5_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����J"
	 ��ۅ��"=*9sequential/densenet169/conv5_block6_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU���J���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����J�"��" *$$1"!*size:512 dest:0 async:1"�  ",`�ե�J؉�"��" *$$1""8�"�  ",W೏�J���"��" *$$1""8�"�  "�����J"
	 �����";*7sequential/densenet169/conv5_block6_1_bn/AssignNewValue"
 �������"  "�  "�����J"
	 �����"=*9sequential/densenet169/conv5_block6_1_bn/AssignNewValue_1"
 �������"  "�  "�໡�J"
	 �܇���"5*1sequential/densenet169/conv5_block6_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "����J"
	 ������"5*1sequential/densenet169/conv5_block6_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����J���"��" *$$1""8�"�  ",a���J���"��" *$$1""8�"�  "���ЅK"
	 ������"5*1sequential/densenet169/conv5_block6_2_conv/Conv2D"
 �������"  "�  "����K"
	 ������"5*1sequential/densenet169/conv5_block6_concat/concat"
 �������"
*output" "*
	 �܇���"
  "  "�  "���БK"
	 �����"=*9sequential/densenet169/conv5_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv5_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[832]"�  "���ŒK"
	 ������"=*9sequential/densenet169/conv5_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[832]"�  "����K"
	 �܇���"=*9sequential/densenet169/conv5_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[832]"�  "�����K"
	 ������"=*9sequential/densenet169/conv5_block7_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[832]"�  "CU�᠔K���"��" *$$1"!*size:3328 dest:0 async:1"�  "CU����K���"��" *$$1"!*size:3328 dest:0 async:1"�  ",`����K���"��" *$$1""8�"�  ",W�ӢK���"��" *$$1""8�"�  "�����K"
	 ��ۅ��";*7sequential/densenet169/conv5_block7_0_bn/AssignNewValue"
 �������"  "�  "��䯧K"
	 �����"=*9sequential/densenet169/conv5_block7_0_bn/AssignNewValue_1"
 �������"  "�  "��ѧ�K"
	 �Ԍ���"5*1sequential/densenet169/conv5_block7_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv5_block7_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv5_block7_1_conv/Conv2D"
 �������"  "�  "���ûK"
	 ������"=*9sequential/densenet169/conv5_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv5_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����K"
	 �����"=*9sequential/densenet169/conv5_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����K"
	 ��ۅ��"=*9sequential/densenet169/conv5_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�𨓽K"
	 ��ۅ��"=*9sequential/densenet169/conv5_block7_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�Ӕ�K���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����K���"��" *$$1"!*size:512 dest:0 async:1"�  ",`Ц��K���"��" *$$1""8�"�  ",W�ܥ�K���"��" *$$1""8�"�  "�����K"
	 �����";*7sequential/densenet169/conv5_block7_1_bn/AssignNewValue"
 �������"  "�  "�����K"
	 �����"=*9sequential/densenet169/conv5_block7_1_bn/AssignNewValue_1"
 �������"  "�  "��ٌ�K"
	 ������"5*1sequential/densenet169/conv5_block7_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�Ѝ��K"
	 �ܚ���"5*1sequential/densenet169/conv5_block7_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R�ʘ�K���"��" *$$1""8�"�  ",a����K���"��" *$$1""8�"�  "�����K"
	 �ܚ���"5*1sequential/densenet169/conv5_block7_2_conv/Conv2D"
 �������"  "�  "�����K"
	 �ܚ���"5*1sequential/densenet169/conv5_block7_concat/concat"
 �������"
*output" "*
	 ������"
  "  "�  "�����K"
	 ��Ę��"=*9sequential/densenet169/conv5_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ��ۅ��"=*9sequential/densenet169/conv5_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[864]"�  "�����K"
	 ������"=*9sequential/densenet169/conv5_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[864]"�  "�����K"
	 ������"=*9sequential/densenet169/conv5_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[864]"�  "�����K"
	 �Ȉ���"=*9sequential/densenet169/conv5_block8_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[864]"�  "CU����K���"��" *$$1"!*size:3456 dest:0 async:1"�  "CU����K���"��" *$$1"!*size:3456 dest:0 async:1"�  ",`���K���"��" *$$1""8�"�  ",W����L���"��" *$$1""8�"�  "�����L"
	 �����";*7sequential/densenet169/conv5_block8_0_bn/AssignNewValue"
 �������"  "�  "���ʅL"
	 ������"=*9sequential/densenet169/conv5_block8_0_bn/AssignNewValue_1"
 �������"  "�  "���L"
	 �����"5*1sequential/densenet169/conv5_block8_1_conv/Conv2D"
 �������"
*output" "*
	 ������"5*1sequential/densenet169/conv5_block8_1_conv/Conv2D"
 �������"*temp" "*
	 ������"5*1sequential/densenet169/conv5_block8_1_conv/Conv2D"
 �������"  "�  "�����L"
	 ������"=*9sequential/densenet169/conv5_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv5_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����L"
	 �����"=*9sequential/densenet169/conv5_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��Ʋ�L"
	 ��ۅ��"=*9sequential/densenet169/conv5_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���ٛL"
	 ��ۅ��"=*9sequential/densenet169/conv5_block8_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�ٜL���"��" *$$1"!*size:512 dest:0 async:1"�  "BU���L���"��" *$$1"!*size:512 dest:0 async:1"�  ",`�֡�L���"��" *$$1""8�"�  ",W����L���"��" *$$1""8�"�  "�Б�L"
	 �����";*7sequential/densenet169/conv5_block8_1_bn/AssignNewValue"
 �������"  "�  "�����L"
	 �����"=*9sequential/densenet169/conv5_block8_1_bn/AssignNewValue_1"
 �������"  "�  "����L"
	 ������"5*1sequential/densenet169/conv5_block8_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "��ʋ�L"
	 ������"5*1sequential/densenet169/conv5_block8_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R�ك�L誘"��" *$$1""8�"�  ",a��ͺL�ʯ"��" *$$1""8�"�  "��ҿL"
	 ������"5*1sequential/densenet169/conv5_block8_2_conv/Conv2D"
 �������"  "�  "�����L"
	 ������"5*1sequential/densenet169/conv5_block8_concat/concat"
 �������"
*output" "*
	 ������"
  "  "�  "�����L"
	 ������"=*9sequential/densenet169/conv5_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������"=*9sequential/densenet169/conv5_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[896]"�  "����L"
	 �����"=*9sequential/densenet169/conv5_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[896]"�  "�����L"
	 �䈖��"=*9sequential/densenet169/conv5_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[896]"�  "�Ȃ��L"
	 ������"=*9sequential/densenet169/conv5_block9_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[896]"�  "CU���L���"��" *$$1"!*size:3584 dest:0 async:1"�  "CU����L���"��" *$$1"!*size:3584 dest:0 async:1"�  ",`����L�ʃ"��" *$$1""8�"�  ",W���L���"��" *$$1""8�"�  "�����L"
	 �����";*7sequential/densenet169/conv5_block9_0_bn/AssignNewValue"
 �������"  "�  "�����L"
	 �悇��"=*9sequential/densenet169/conv5_block9_0_bn/AssignNewValue_1"
 �������"  "�  "�����L"
	 ��ә��"5*1sequential/densenet169/conv5_block9_1_conv/Conv2D"
 �������"
*output" "*
	 ��ٙ��"5*1sequential/densenet169/conv5_block9_1_conv/Conv2D"
 �������"*temp" "*
	 ��ٙ��"5*1sequential/densenet169/conv5_block9_1_conv/Conv2D"
 �������"  "�  "�����L"
	 ��ٙ��"=*9sequential/densenet169/conv5_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����"=*9sequential/densenet169/conv5_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�І��L"
	 �����"=*9sequential/densenet169/conv5_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����L"
	 �����"=*9sequential/densenet169/conv5_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���L"
	 �����"=*9sequential/densenet169/conv5_block9_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����L���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����L���"��" *$$1"!*size:512 dest:0 async:1"�  ",`��M�Ό"��" *$$1""8�"�  ",W�Ԓ�M���"��" *$$1""8�"�  "��ȗ�M"
	 �����";*7sequential/densenet169/conv5_block9_1_bn/AssignNewValue"
 �������"  "�  "�����M"
	 �����"=*9sequential/densenet169/conv5_block9_1_bn/AssignNewValue_1"
 �������"  "�  "���ƏM"
	 ������"5*1sequential/densenet169/conv5_block9_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "���ڐM"
	 ��ߙ��"5*1sequential/densenet169/conv5_block9_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R��ԑM���"��" *$$1""8�"�  ",a���M�ӹ"��" *$$1""8�"�  "���ҞM"
	 ��ߙ��"5*1sequential/densenet169/conv5_block9_2_conv/Conv2D"
 �������"  "�  "����M"
	 ��ߙ��"5*1sequential/densenet169/conv5_block9_concat/concat"
 �������"
*output" "*
	 ������"
  "  "�  "���ԪM"
	 ����">*:sequential/densenet169/conv5_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����">*:sequential/densenet169/conv5_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[928]"�  "���̫M"
	 ������">*:sequential/densenet169/conv5_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[928]"�  "�����M"
	 ������">*:sequential/densenet169/conv5_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[928]"�  "��П�M"
	 �Й���">*:sequential/densenet169/conv5_block10_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[928]"�  "CUЧ��M���"��" *$$1"!*size:3712 dest:0 async:1"�  "CU�瘱M�Ϟ"��" *$$1"!*size:3712 dest:0 async:1"�  ",`����MȈ�"��" *$$1""8�"�  ",W����M���"��" *$$1""8�"�  "���׿M"
	 �����"<*8sequential/densenet169/conv5_block10_0_bn/AssignNewValue"
 �������"  "�  "�����M"
	 ������">*:sequential/densenet169/conv5_block10_0_bn/AssignNewValue_1"
 �������"  "�  "�����M"
	 ������"6*2sequential/densenet169/conv5_block10_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block10_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv5_block10_1_conv/Conv2D"
 �������"  "�  "�轟�M"
	 ������">*:sequential/densenet169/conv5_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �����">*:sequential/densenet169/conv5_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�Ȍ��M"
	 �����">*:sequential/densenet169/conv5_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����M"
	 �����">*:sequential/densenet169/conv5_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����M"
	 �����">*:sequential/densenet169/conv5_block10_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BUА��M���"��" *$$1"!*size:512 dest:0 async:1"�  "BU�؉�M���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����M���"��" *$$1""8�"�  ",W����M���"��" *$$1""8�"�  "�����M"
	 ������"<*8sequential/densenet169/conv5_block10_1_bn/AssignNewValue"
 �������"  "�  "�����M"
	 ������">*:sequential/densenet169/conv5_block10_1_bn/AssignNewValue_1"
 �������"  "�  "�����M"
	 ��ƚ��"6*2sequential/densenet169/conv5_block10_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����M"
	 ��Ț��"6*2sequential/densenet169/conv5_block10_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����M�ğ"��" *$$1""8�"�  ",a����M��"��" *$$1""8�"�  "�����M"
	 ��Ț��"6*2sequential/densenet169/conv5_block10_2_conv/Conv2D"
 �������"  "�  "�����N"
	 ��Ț��"6*2sequential/densenet169/conv5_block10_concat/concat"
 �������"
*output" "*
	 ��ƚ��"
  "  "�  "�����N"
	 ������">*:sequential/densenet169/conv5_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[960]"�  "�����N"
	 �����">*:sequential/densenet169/conv5_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[960]"�  "�����N"
	 ���">*:sequential/densenet169/conv5_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[960]"�  "���ӊN"
	 ������">*:sequential/densenet169/conv5_block11_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[960]"�  "CU���N���"��" *$$1"!*size:3840 dest:0 async:1"�  "CU���N���"��" *$$1"!*size:3840 dest:0 async:1"�  ",`Д��N���"��" *$$1""8�"�  ",W��ÛN�ڃ"��" *$$1""8�"�  "�����N"
	 ������"<*8sequential/densenet169/conv5_block11_0_bn/AssignNewValue"
 �������"  "�  "�ॶ�N"
	 ������">*:sequential/densenet169/conv5_block11_0_bn/AssignNewValue_1"
 �������"  "�  "�����N"
	 ������"6*2sequential/densenet169/conv5_block11_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block11_1_conv/Conv2D"
 �������"*temp" "*
	 ������"6*2sequential/densenet169/conv5_block11_1_conv/Conv2D"
 �������"  "�  "�����N"
	 ������">*:sequential/densenet169/conv5_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ɾ�N"
	 ������">*:sequential/densenet169/conv5_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����N"
	 �����">*:sequential/densenet169/conv5_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�𙏷N"
	 �����">*:sequential/densenet169/conv5_block11_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����N���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����N���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����N�ǌ"��" *$$1""8�"�  ",W����N��"��" *$$1""8�"�  "�����N"
	 ����"<*8sequential/densenet169/conv5_block11_1_bn/AssignNewValue"
 �������"  "�  "�����N"
	 ������">*:sequential/densenet169/conv5_block11_1_bn/AssignNewValue_1"
 �������"  "�  "�����N"
	 ��ƚ��"6*2sequential/densenet169/conv5_block11_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�؋��N"
	 �Ĳ���"6*2sequential/densenet169/conv5_block11_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����N���"��" *$$1""8�"�  ",a�Ԩ�NА�"��" *$$1""8�"�  "����N"
	 �Ĳ���"6*2sequential/densenet169/conv5_block11_2_conv/Conv2D"
 �������"  "�  "�����N"
	 �Ĳ���"6*2sequential/densenet169/conv5_block11_concat/concat"
 �������"
*output" "*
	 ��ƚ��"
  "  "�  "��΁�N"
	 �����">*:sequential/densenet169/conv5_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[992]"�  "�����N"
	 ������">*:sequential/densenet169/conv5_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[992]"�  "�����N"
	 ��ƚ��">*:sequential/densenet169/conv5_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[992]"�  "�����N"
	 ��ƚ��">*:sequential/densenet169/conv5_block12_0_bn/FusedBatchNormV3"
 �������"
*output" "	*[992]"�  "CU����Nض�"��" *$$1"!*size:3968 dest:0 async:1"�  "CU����N�և"��" *$$1"!*size:3968 dest:0 async:1"�  ",`����N���"��" *$$1""8�"�  ",W����N��"��" *$$1""8�"�  "��ڔ�N"
	 ������"<*8sequential/densenet169/conv5_block12_0_bn/AssignNewValue"
 �������"  "�  "�����N"
	 ������">*:sequential/densenet169/conv5_block12_0_bn/AssignNewValue_1"
 �������"  "�  "�����O"
	 ������"6*2sequential/densenet169/conv5_block12_1_conv/Conv2D"
 �������"
*output" "*
	 �ԙ���"6*2sequential/densenet169/conv5_block12_1_conv/Conv2D"
 �������"*temp" "*
	 �ԙ���"6*2sequential/densenet169/conv5_block12_1_conv/Conv2D"
 �������"  "�  "��ԃ�O"
	 �ԙ���">*:sequential/densenet169/conv5_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ����">*:sequential/densenet169/conv5_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����O"
	 ������">*:sequential/densenet169/conv5_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����O"
	 �悇��">*:sequential/densenet169/conv5_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��֕O"
	 �ꂇ��">*:sequential/densenet169/conv5_block12_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�ܖO�ӏ"��" *$$1"!*size:512 dest:0 async:1"�  "BU����O�؎"��" *$$1"!*size:512 dest:0 async:1"�  ",`Ȉ�O��"��" *$$1""8�"�  ",W����OЭ�"��" *$$1""8�"�  "��˲�O"
	 ������"<*8sequential/densenet169/conv5_block12_1_bn/AssignNewValue"
 �������"  "�  "���ߩO"
	 ������">*:sequential/densenet169/conv5_block12_1_bn/AssignNewValue_1"
 �������"  "�  "�؂ޫO"
	 ������"6*2sequential/densenet169/conv5_block12_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����O"
	 ������"6*2sequential/densenet169/conv5_block12_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R���O�ѓ"��" *$$1""8�"�  ",a����O��"��" *$$1""8�"�  "��ྺO"
	 ������"6*2sequential/densenet169/conv5_block12_2_conv/Conv2D"
 �������"  "�  "���߽O"
	 ������"6*2sequential/densenet169/conv5_block12_concat/concat"
 �������"
*output" "*[200,1024,1,1]"�  ",N����O���"��" *$$1""8�"�  ",N����O�׻"��" *$$1""8�"�  "w���O"
	 ������"
  "  "�  "�����O"
	 ��Ӝ��">*:sequential/densenet169/conv5_block13_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1024,1,1]"�  "��ʔ�O"
	 ������">*:sequential/densenet169/conv5_block13_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1024]"�  "�����O"
	 ������">*:sequential/densenet169/conv5_block13_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1024]"�  "�����O"
	 ��ǚ��">*:sequential/densenet169/conv5_block13_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1024]"�  "�����O"
	 ��ǚ��">*:sequential/densenet169/conv5_block13_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1024]"�  "CU�ɫ�O���"��" *$$1"!*size:4096 dest:0 async:1"�  "CU����O���"��" *$$1"!*size:4096 dest:0 async:1"�  ",`����O���"��" *$$1""8�"�  ",W�Ŷ�O��"��" *$$1""8�"�  "���O"
	 �Ȫ���"<*8sequential/densenet169/conv5_block13_0_bn/AssignNewValue"
 �������"  "�  "��ڋ�O"
	 ������">*:sequential/densenet169/conv5_block13_0_bn/AssignNewValue_1"
 �������"  "�  "�����O"
	 ������"6*2sequential/densenet169/conv5_block13_1_conv/Conv2D"
 �������"
*output" "*
	 �܋���"6*2sequential/densenet169/conv5_block13_1_conv/Conv2D"
 �������"*temp" "*[128,1024,1,1]"�  ",R����O�Ґ"��" *$$1""8�"�  ",a����O���"��" *$$1""8�"�  "�����O"
	 �܋���"6*2sequential/densenet169/conv5_block13_1_conv/Conv2D"
 �������"  "�  "����O"
	 �܋���">*:sequential/densenet169/conv5_block13_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block13_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ʴ�O"
	 ������">*:sequential/densenet169/conv5_block13_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����O"
	 ���">*:sequential/densenet169/conv5_block13_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����O"
	 ����">*:sequential/densenet169/conv5_block13_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU���O���"��" *$$1"!*size:512 dest:0 async:1"�  "BU���OЀ�"��" *$$1"!*size:512 dest:0 async:1"�  ",`����O���"��" *$$1""8�"�  ",W���P���"��" *$$1""8�"�  "�����P"
	 ������"<*8sequential/densenet169/conv5_block13_1_bn/AssignNewValue"
 �������"  "�  "���ӇP"
	 ������">*:sequential/densenet169/conv5_block13_1_bn/AssignNewValue_1"
 �������"  "�  "���ˉP"
	 ������"6*2sequential/densenet169/conv5_block13_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "���ڊP"
	 ������"6*2sequential/densenet169/conv5_block13_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R��ԋP�а"��" *$$1""8�"�  ",a��P���"��" *$$1""8�"�  "���ߘP"
	 ������"6*2sequential/densenet169/conv5_block13_2_conv/Conv2D"
 �������"  "�  "�ػ�P"
	 ������"6*2sequential/densenet169/conv5_block13_concat/concat"
 �������"
*output" "*[200,1056,1,1]"�  ",NȜ��P���"��" *$$1""8�"�  ",N�ǠP���"��" *$$1""8�"�  "w��ޢP"
	 ������"
  "  "�  "�����P"
	 ��ŝ��">*:sequential/densenet169/conv5_block14_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1056,1,1]"�  "���ޥP"
	 �Ȫ���">*:sequential/densenet169/conv5_block14_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1056]"�  "�ૐ�P"
	 ��ǚ��">*:sequential/densenet169/conv5_block14_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1056]"�  "��ẦP"
*�~��?" ����" �!" �&"
	 ��ǚ��">*:sequential/densenet169/conv5_block14_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1056]"�  "����P"
*�~��?" ����" �!" �""
	 ������">*:sequential/densenet169/conv5_block14_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1056]"�  "CU���P���"��" *$$1"!*size:4224 dest:0 async:1"�  "CUط��P���"��" *$$1"!*size:4224 dest:0 async:1"�  ",`��ɰP���"��" *$$1""8�"�  ",W����P�ڃ"��" *$$1""8�"�  "�����P"
	 �ث���"<*8sequential/densenet169/conv5_block14_0_bn/AssignNewValue"
 �������"  "�  "�����P"
	 ������">*:sequential/densenet169/conv5_block14_0_bn/AssignNewValue_1"
 �������"  "�  "��ӣ�P"
	 ������"6*2sequential/densenet169/conv5_block14_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block14_1_conv/Conv2D"
 �������"*temp" "*[128,1056,1,1]"�  ",R����P���"��" *$$1""8�"�  ",a����P���"��" *$$1""8�"�  "�����P"
	 ������"6*2sequential/densenet169/conv5_block14_1_conv/Conv2D"
 �������"  "�  "�����P"
	 ������">*:sequential/densenet169/conv5_block14_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block14_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�Ы��P"
	 ������">*:sequential/densenet169/conv5_block14_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����P"
	 ������">*:sequential/densenet169/conv5_block14_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����P"
	 ������">*:sequential/densenet169/conv5_block14_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����P���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����P���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����P���"��" *$$1""8�"�  ",W���PЌ�"��" *$$1""8�"�  "�����P"
	 ������"<*8sequential/densenet169/conv5_block14_1_bn/AssignNewValue"
 �������"  "�  "�����P"
	 ������">*:sequential/densenet169/conv5_block14_1_bn/AssignNewValue_1"
 �������"  "�  "��ƹ�P"
	 �̅���"6*2sequential/densenet169/conv5_block14_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����P"
	 ������"6*2sequential/densenet169/conv5_block14_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����P���"��" *$$1""8�"�  ",a����P�׵"��" *$$1""8�"�  "��ǖ�P"
	 ������"6*2sequential/densenet169/conv5_block14_2_conv/Conv2D"
 �������"  "�  "�Е��P"
	 ������"6*2sequential/densenet169/conv5_block14_concat/concat"
 �������"
*output" "*[200,1088,1,1]"�  ",N���PȺ�"��" *$$1""8�"�  ",N�θ�P���"��" *$$1""8�"�  "w����P"
	 �̅���"
  "  "�  "�����Q"
	 ������">*:sequential/densenet169/conv5_block15_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1088,1,1]"�  "���قQ"
	 ������">*:sequential/densenet169/conv5_block15_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1088]"�  "��Q"
	 �ث���">*:sequential/densenet169/conv5_block15_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1088]"�  "�����Q"
	 ������">*:sequential/densenet169/conv5_block15_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1088]"�  "���߃Q"
	 ������">*:sequential/densenet169/conv5_block15_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1088]"�  "CU����Q���"��" *$$1"!*size:4352 dest:0 async:1"�  "CU���Q���"��" *$$1"!*size:4352 dest:0 async:1"�  ",`����Q���"��" *$$1""8�"�  ",W�ϧ�Q�Ӏ"��" *$$1""8�"�  "�����Q"
	 �ꬑ��"<*8sequential/densenet169/conv5_block15_0_bn/AssignNewValue"
 �������"  "�  "��͓�Q"
	 ������">*:sequential/densenet169/conv5_block15_0_bn/AssignNewValue_1"
 �������"  "�  "��뀜Q"
	 �����"6*2sequential/densenet169/conv5_block15_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block15_1_conv/Conv2D"
 �������"*temp" "*[128,1088,1,1]"�  ",Rإ��Q���"��" *$$1""8�"�  ",aȐ�Q���"��" *$$1""8�"�  "�����Q"
	 ������"6*2sequential/densenet169/conv5_block15_1_conv/Conv2D"
 �������"  "�  "�����Q"
	 ������">*:sequential/densenet169/conv5_block15_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block15_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�ز��Q"
	 ������">*:sequential/densenet169/conv5_block15_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���֯Q"
	 ������">*:sequential/densenet169/conv5_block15_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����Q"
	 ������">*:sequential/densenet169/conv5_block15_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����Q���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����Q���"��" *$$1"!*size:512 dest:0 async:1"�  ",`��չQ���"��" *$$1""8�"�  ",W����QЉ�"��" *$$1""8�"�  "����Q"
	 ������"<*8sequential/densenet169/conv5_block15_1_bn/AssignNewValue"
 �������"  "�  "�����Q"
	 ������">*:sequential/densenet169/conv5_block15_1_bn/AssignNewValue_1"
 �������"  "�  "��ۤ�Q"
	 �̅���"6*2sequential/densenet169/conv5_block15_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "����Q"
	 ������"6*2sequential/densenet169/conv5_block15_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����Q�ԟ"��" *$$1""8�"�  ",a�ٌ�Q���"��" *$$1""8�"�  "�؆��Q"
	 ������"6*2sequential/densenet169/conv5_block15_2_conv/Conv2D"
 �������"  "�  "����Q"
	 ������"6*2sequential/densenet169/conv5_block15_concat/concat"
 �������"
*output" "*[200,1120,1,1]"�  ",N����Q��"��" *$$1""8�"�  ",N����Q���"��" *$$1""8�"�  "w����Q"
	 �̅���"
  "  "�  "�����Q"
	 �̴���">*:sequential/densenet169/conv5_block16_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1120,1,1]"�  "�蚹�Q"
	 �ꬑ��">*:sequential/densenet169/conv5_block16_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1120]"�  "�����Q"
	 �ڠ���">*:sequential/densenet169/conv5_block16_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1120]"�  "��۞�Q"
	 ������">*:sequential/densenet169/conv5_block16_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1120]"�  "�����Q"
	 �̅���">*:sequential/densenet169/conv5_block16_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1120]"�  "CU����Q���"��" *$$1"!*size:4480 dest:0 async:1"�  "CU����Qఉ"��" *$$1"!*size:4480 dest:0 async:1"�  ",`舞�Q��"��" *$$1""8�"�  ",W���Q���"��" *$$1""8�"�  "�����Q"
	 ������"<*8sequential/densenet169/conv5_block16_0_bn/AssignNewValue"
 �������"  "�  "�ॐ�Q"
	 ������">*:sequential/densenet169/conv5_block16_0_bn/AssignNewValue_1"
 �������"  "�  "�����Q"
	 ������"6*2sequential/densenet169/conv5_block16_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block16_1_conv/Conv2D"
 �������"*temp" "*[128,1120,1,1]"�  ",R�ߧ�Q���"��" *$$1""8�"�  ",aȎ��R���"��" *$$1""8�"�  "��ͺ�R"
	 ������"6*2sequential/densenet169/conv5_block16_1_conv/Conv2D"
 �������"  "�  "���܋R"
	 ������">*:sequential/densenet169/conv5_block16_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block16_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���܌R"
	 ������">*:sequential/densenet169/conv5_block16_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����R"
	 ������">*:sequential/densenet169/conv5_block16_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��گ�R"
	 ������">*:sequential/densenet169/conv5_block16_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�Ъ�R���"��" *$$1"!*size:512 dest:0 async:1"�  "BU�漣R�ɓ"��" *$$1"!*size:512 dest:0 async:1"�  ",`ؗ��R���"��" *$$1""8�"�  ",W���R���"��" *$$1""8�"�  "�ث��R"
	 ������"<*8sequential/densenet169/conv5_block16_1_bn/AssignNewValue"
 �������"  "�  "��ǖ�R"
	 ������">*:sequential/densenet169/conv5_block16_1_bn/AssignNewValue_1"
 �������"  "�  "��㞣R"
	 ������"6*2sequential/densenet169/conv5_block16_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����R"
	 ������"6*2sequential/densenet169/conv5_block16_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R�Χ�R���"��" *$$1""8�"�  ",a����R���"��" *$$1""8�"�  "���ʲR"
	 ������"6*2sequential/densenet169/conv5_block16_2_conv/Conv2D"
 �������"  "�  "�����R"
	 ������"6*2sequential/densenet169/conv5_block16_concat/concat"
 �������"
*output" "*[200,1152,1,1]"�  ",N����R���"��" *$$1""8�"�  ",N����R���"��" *$$1""8�"�  "wș��R"
	 ������"
  "  "�  "��۾R"
	 ��Ơ��">*:sequential/densenet169/conv5_block17_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1152,1,1]"�  "��ܥ�R"
���?" ����" �$" �$"
	 ������">*:sequential/densenet169/conv5_block17_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1152]"�  "����R"
	 ������">*:sequential/densenet169/conv5_block17_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1152]"�  "����R"
	 ������">*:sequential/densenet169/conv5_block17_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1152]"�  "����R"
	 ������">*:sequential/densenet169/conv5_block17_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1152]"�  "CU����R��"��" *$$1"!*size:4608 dest:0 async:1"�  "CU�շ�R�"��" *$$1"!*size:4608 dest:0 async:1"�  ",`����R؆�"��" *$$1""8�"�  ",W����R��"��" *$$1""8�"�  "�����R"
	 ������"<*8sequential/densenet169/conv5_block17_0_bn/AssignNewValue"
 �������"  "�  "�����R"
	 �Ʈ���">*:sequential/densenet169/conv5_block17_0_bn/AssignNewValue_1"
 �������"  "�  "�����R"
	 ������"6*2sequential/densenet169/conv5_block17_1_conv/Conv2D"
 �������"
*output" "*
	 �脡��"6*2sequential/densenet169/conv5_block17_1_conv/Conv2D"
 �������"*temp" "*[128,1152,1,1]"�  ",R����R���"��" *$$1""8�"�  ",aȤ��R���"��" *$$1""8�"�  "�����R"
	 �脡��"6*2sequential/densenet169/conv5_block17_1_conv/Conv2D"
 �������"  "�  "�����R"
	 �脡��">*:sequential/densenet169/conv5_block17_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block17_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����R"
	 ������">*:sequential/densenet169/conv5_block17_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����R"
	 ������">*:sequential/densenet169/conv5_block17_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ʻ�R"
	 ������">*:sequential/densenet169/conv5_block17_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU؛��R���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����R���"��" *$$1"!*size:512 dest:0 async:1"�  ",`Ⱦ��R���"��" *$$1""8�"�  ",W����R���"��" *$$1""8�"�  "��Ά�R"
	 ������"<*8sequential/densenet169/conv5_block17_1_bn/AssignNewValue"
 �������"  "�  "��矀S"
	 ������">*:sequential/densenet169/conv5_block17_1_bn/AssignNewValue_1"
 �������"  "�  "�����S"
	 ������"6*2sequential/densenet169/conv5_block17_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "��ũ�S"
	 ������"6*2sequential/densenet169/conv5_block17_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R�Ŧ�S���"��" *$$1""8�"�  ",a����SȄ�"��" *$$1""8�"�  "��ؓ�S"
	 ������"6*2sequential/densenet169/conv5_block17_2_conv/Conv2D"
 �������"  "�  "�����S"
	 ������"6*2sequential/densenet169/conv5_block17_concat/concat"
 �������"
*output" "*[200,1184,1,1]"�  ",N�Ӻ�S�˯"��" *$$1""8�"�  ",N����S���"��" *$$1""8�"�  "w��S"
	 ������"
  "  "�  "��ؖ�S"
	 ��ġ��">*:sequential/densenet169/conv5_block18_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1184,1,1]"�  "����S"
	 ������">*:sequential/densenet169/conv5_block18_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1184]"�  "�����S"
	 ������">*:sequential/densenet169/conv5_block18_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1184]"�  "����S"
	 �ކ���">*:sequential/densenet169/conv5_block18_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1184]"�  "�𷠟S"
	 ������">*:sequential/densenet169/conv5_block18_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1184]"�  "CU��ؠS���"��" *$$1"!*size:4736 dest:0 async:1"�  "CU��դS���"��" *$$1"!*size:4736 dest:0 async:1"�  ",`譟�S���"��" *$$1""8�"�  ",W����S���"��" *$$1""8�"�  "��ǐ�S"
	 ������"<*8sequential/densenet169/conv5_block18_0_bn/AssignNewValue"
 �������"  "�  "�؂��S"
	 ������">*:sequential/densenet169/conv5_block18_0_bn/AssignNewValue_1"
 �������"  "�  "�����S"
	 ������"6*2sequential/densenet169/conv5_block18_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block18_1_conv/Conv2D"
 �������"*temp" "*[128,1184,1,1]"�  ",R���S���"��" *$$1""8�"�  ",a����S���"��" *$$1""8�"�  "�����S"
	 ������"6*2sequential/densenet169/conv5_block18_1_conv/Conv2D"
 �������"  "�  "�����S"
	 ������">*:sequential/densenet169/conv5_block18_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block18_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����S"
	 ������">*:sequential/densenet169/conv5_block18_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����S"
	 ������">*:sequential/densenet169/conv5_block18_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ͽ�S"
	 ������">*:sequential/densenet169/conv5_block18_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BUȱ��S���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����S�ˍ"��" *$$1"!*size:512 dest:0 async:1"�  ",`����S���"��" *$$1""8�"�  ",W����SȌ�"��" *$$1""8�"�  "�����S"
	 ������"<*8sequential/densenet169/conv5_block18_1_bn/AssignNewValue"
 �������"  "�  "��͓�S"
	 ������">*:sequential/densenet169/conv5_block18_1_bn/AssignNewValue_1"
 �������"  "�  "�ே�S"
	 ������"6*2sequential/densenet169/conv5_block18_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����S"
	 ������"6*2sequential/densenet169/conv5_block18_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����Sؠ�"��" *$$1""8�"�  ",a����S�ߥ"��" *$$1""8�"�  "�����S"
	 ������"6*2sequential/densenet169/conv5_block18_2_conv/Conv2D"
 �������"  "�  "�����S"
	 ������"6*2sequential/densenet169/conv5_block18_concat/concat"
 �������"
*output" "*[200,1216,1,1]"�  ",Nئ��S���"��" *$$1""8�"�  ",N����S���"��" *$$1""8�"�  "w����S"
	 ������"
  "  "�  "��٧�T"
	 ��Ȣ��">*:sequential/densenet169/conv5_block19_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1216,1,1]"�  "�����T"
	 ������">*:sequential/densenet169/conv5_block19_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1216]"�  "��î�T"
	 ������">*:sequential/densenet169/conv5_block19_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1216]"�  "���܃T"
	 �挠��">*:sequential/densenet169/conv5_block19_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1216]"�  "�����T"
	 ������">*:sequential/densenet169/conv5_block19_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1216]"�  "CU�箅T�۾"��" *$$1"!*size:4864 dest:0 async:1"�  "CU����T��"��" *$$1"!*size:4864 dest:0 async:1"�  ",`�ˏ�TЃ�"��" *$$1""8�"�  ",WУەT�"��" *$$1""8�"�  "���șT"
	 ������"<*8sequential/densenet169/conv5_block19_0_bn/AssignNewValue"
 �������"  "�  "����T"
	 �ڄ���">*:sequential/densenet169/conv5_block19_0_bn/AssignNewValue_1"
 �������"  "�  "����T"
	 ������"6*2sequential/densenet169/conv5_block19_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block19_1_conv/Conv2D"
 �������"*temp" "*[128,1216,1,1]"�  ",R����T���"��" *$$1""8�"�  ",a�ڕ�T���"��" *$$1""8�"�  "���ܭT"
	 ������"6*2sequential/densenet169/conv5_block19_1_conv/Conv2D"
 �������"  "�  "�؏��T"
	 ������">*:sequential/densenet169/conv5_block19_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block19_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����T"
	 ������">*:sequential/densenet169/conv5_block19_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����T"
	 ������">*:sequential/densenet169/conv5_block19_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����T"
	 ������">*:sequential/densenet169/conv5_block19_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����T���"��" *$$1"!*size:512 dest:0 async:1"�  "BU�끸T���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����T�ˍ"��" *$$1""8�"�  ",W�ծ�T���"��" *$$1""8�"�  "�����T"
	 ������"<*8sequential/densenet169/conv5_block19_1_bn/AssignNewValue"
 �������"  "�  "��Η�T"
	 ������">*:sequential/densenet169/conv5_block19_1_bn/AssignNewValue_1"
 �������"  "�  "��͡�T"
	 ������"6*2sequential/densenet169/conv5_block19_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "��ʴ�T"
	 ������"6*2sequential/densenet169/conv5_block19_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R�ԫ�T���"��" *$$1""8�"�  ",aЉ��T���"��" *$$1""8�"�  "�Р��T"
	 ������"6*2sequential/densenet169/conv5_block19_2_conv/Conv2D"
 �������"  "�  "�����T"
	 ������"6*2sequential/densenet169/conv5_block19_concat/concat"
 �������"
*output" "*[200,1248,1,1]"�  ",N����T�˼"��" *$$1""8�"�  ",N����T���"��" *$$1""8�"�  "w����T"
	 ������"
  "  "�  "�����T"
	 ��̣��">*:sequential/densenet169/conv5_block20_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1248,1,1]"�  "�����T"
	 ������">*:sequential/densenet169/conv5_block20_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1248]"�  "�����T"
	 ������">*:sequential/densenet169/conv5_block20_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1248]"�  "�����T"
	 �ڍ���">*:sequential/densenet169/conv5_block20_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1248]"�  "�����T"
	 ������">*:sequential/densenet169/conv5_block20_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1248]"�  "CU���TȰ�"��" *$$1"!*size:4992 dest:0 async:1"�  "CU����T���"��" *$$1"!*size:4992 dest:0 async:1"�  ",`����T���"��" *$$1""8�"�  ",W����T���"��" *$$1""8�"�  "�����T"
	 ������"<*8sequential/densenet169/conv5_block20_0_bn/AssignNewValue"
 �������"  "�  "��ݹ�T"
	 �����">*:sequential/densenet169/conv5_block20_0_bn/AssignNewValue_1"
 �������"  "�  "�����T"
	 ������"6*2sequential/densenet169/conv5_block20_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block20_1_conv/Conv2D"
 �������"*temp" "*[128,1248,1,1]"�  ",R����U�ؽ"��" *$$1""8�"�  ",a𣧋U���"��" *$$1""8�"�  "����U"
	 ������"6*2sequential/densenet169/conv5_block20_1_conv/Conv2D"
 �������"  "�  "�����U"
	 ������">*:sequential/densenet169/conv5_block20_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block20_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��՘�U"
	 ������">*:sequential/densenet169/conv5_block20_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���U"
	 ������">*:sequential/densenet169/conv5_block20_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����U"
	 ������">*:sequential/densenet169/conv5_block20_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU���U���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����U���"��" *$$1"!*size:512 dest:0 async:1"�  ",`Ț��U���"��" *$$1""8�"�  ",W����U���"��" *$$1""8�"�  "��ă�U"
	 �ֆ���"<*8sequential/densenet169/conv5_block20_1_bn/AssignNewValue"
 �������"  "�  "��ʥ�U"
	 �چ���">*:sequential/densenet169/conv5_block20_1_bn/AssignNewValue_1"
 �������"  "�  "���ĭU"
	 ������"6*2sequential/densenet169/conv5_block20_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�؍¯U"
	 ������"6*2sequential/densenet169/conv5_block20_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R��ʰU���"��" *$$1""8�"�  ",a����Uإ�"��" *$$1""8�"�  "�����U"
	 ������"6*2sequential/densenet169/conv5_block20_2_conv/Conv2D"
 �������"  "�  "�Ⱥ��U"
	 ������"6*2sequential/densenet169/conv5_block20_concat/concat"
 �������"
*output" "*[200,1280,1,1]"�  ",N���U���"��" *$$1""8�"�  ",N����U���"��" *$$1""8�"�  "w螲�U"
	 ������"
  "  "�  "�Љ��U"
��?" ����" ��>" ��>"
	 ��֤��">*:sequential/densenet169/conv5_block21_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1280,1,1]"�  "�����U"
��?" ����" �(" �("
	 �����">*:sequential/densenet169/conv5_block21_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1280]"�  "�����U"
	 ������">*:sequential/densenet169/conv5_block21_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1280]"�  "�����U"
	 ������">*:sequential/densenet169/conv5_block21_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1280]"�  "�賜�U"
	 �苢��">*:sequential/densenet169/conv5_block21_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1280]"�  "CU�پ�U���"��" *$$1"!*size:5120 dest:0 async:1"�  "CU����U���"��" *$$1"!*size:5120 dest:0 async:1"�  ",`����U���"��" *$$1""8�"�  ",Wȧ��U�Ё"��" *$$1""8�"�  "����U"
	 ������"<*8sequential/densenet169/conv5_block21_0_bn/AssignNewValue"
 �������"  "�  "��ѱ�U"
	 �ԇ���">*:sequential/densenet169/conv5_block21_0_bn/AssignNewValue_1"
 �������"  "�  "�����U"
	 ������"6*2sequential/densenet169/conv5_block21_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block21_1_conv/Conv2D"
 �������"*temp" "*[128,1280,1,1]"�  ",R�ܖ�U���"��" *$$1""8�"�  ",a���U���"��" *$$1""8�"�  "�����U"
	 ������"6*2sequential/densenet169/conv5_block21_1_conv/Conv2D"
 �������"  "�  "��֊�U"
	 ������">*:sequential/densenet169/conv5_block21_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �ֆ���">*:sequential/densenet169/conv5_block21_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����U"
	 �چ���">*:sequential/densenet169/conv5_block21_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����U"
	 ������">*:sequential/densenet169/conv5_block21_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����U"
	 ������">*:sequential/densenet169/conv5_block21_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����U��"��" *$$1"!*size:512 dest:0 async:1"�  "BU���U���"��" *$$1"!*size:512 dest:0 async:1"�  ",`�ᡄV���"��" *$$1""8�"�  ",W����V�"��" *$$1""8�"�  "���ߍV"
	 ������"<*8sequential/densenet169/conv5_block21_1_bn/AssignNewValue"
 �������"  "�  "����V"
	 ������">*:sequential/densenet169/conv5_block21_1_bn/AssignNewValue_1"
 �������"  "�  "���ӑV"
	 ������"6*2sequential/densenet169/conv5_block21_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "����V"
	 ������"6*2sequential/densenet169/conv5_block21_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R���V൱"��" *$$1""8�"�  ",a���V���"��" *$$1""8�"�  "�����V"
	 ������"6*2sequential/densenet169/conv5_block21_2_conv/Conv2D"
 �������"  "�  "��٬�V"
	 ������"6*2sequential/densenet169/conv5_block21_concat/concat"
 �������"
*output" "*[200,1312,1,1]"�  ",N����V���"��" *$$1""8�"�  ",Nț٨V���"��" *$$1""8�"�  "w���V"
	 ������"
  "  "�  "�����V"
	 �����">*:sequential/densenet169/conv5_block22_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1312,1,1]"�  "���ݭV"
	 ������">*:sequential/densenet169/conv5_block22_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1312]"�  "���ޮV"
	 ������">*:sequential/densenet169/conv5_block22_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1312]"�  "��ז�V"
	 ������">*:sequential/densenet169/conv5_block22_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1312]"�  "��ȯV"
	 �ږ���">*:sequential/densenet169/conv5_block22_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1312]"�  "CU���V���"��" *$$1"!*size:5248 dest:0 async:1"�  "CU����V���"��" *$$1"!*size:5248 dest:0 async:1"�  ",`ЯֹV�ʋ"��" *$$1""8�"�  ",W��տV���"��" *$$1""8�"�  "�،��V"
	 �܈���"<*8sequential/densenet169/conv5_block22_0_bn/AssignNewValue"
 �������"  "�  "�����V"
	 ������">*:sequential/densenet169/conv5_block22_0_bn/AssignNewValue_1"
 �������"  "�  "�����V"
	 �ȡ���"6*2sequential/densenet169/conv5_block22_1_conv/Conv2D"
 �������"
*output" "*
	 �触��"6*2sequential/densenet169/conv5_block22_1_conv/Conv2D"
 �������"*temp" "*[128,1312,1,1]"�  ",R����Vȕ�"��" *$$1""8�"�  ",a؝��VЙ�"��" *$$1""8�"�  "����V"
	 �触��"6*2sequential/densenet169/conv5_block22_1_conv/Conv2D"
 �������"  "�  "�؉��V"
	 �触��">*:sequential/densenet169/conv5_block22_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block22_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����V"
	 ������">*:sequential/densenet169/conv5_block22_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��݊�V"
	 ������">*:sequential/densenet169/conv5_block22_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����V"
	 ������">*:sequential/densenet169/conv5_block22_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU���V���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����V���"��" *$$1"!*size:512 dest:0 async:1"�  ",`�ї�V���"��" *$$1""8�"�  ",W����V���"��" *$$1""8�"�  "�����V"
	 �܉���"<*8sequential/densenet169/conv5_block22_1_bn/AssignNewValue"
 �������"  "�  "�����V"
	 ������">*:sequential/densenet169/conv5_block22_1_bn/AssignNewValue_1"
 �������"  "�  "�����V"
	 ������"6*2sequential/densenet169/conv5_block22_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����V"
	 �Я���"6*2sequential/densenet169/conv5_block22_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����V���"��" *$$1""8�"�  ",a����V�ͻ"��" *$$1""8�"�  "�����W"
	 �Я���"6*2sequential/densenet169/conv5_block22_2_conv/Conv2D"
 �������"  "�  "��୅W"
	 �Я���"6*2sequential/densenet169/conv5_block22_concat/concat"
 �������"
*output" "*[200,1344,1,1]"�  ",N��ǆW���"��" *$$1""8�"�  ",N�͒�W���"��" *$$1""8�"�  "w��W"
	 ������"
  "  "�  "�����W"
	 �����">*:sequential/densenet169/conv5_block23_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1344,1,1]"�  "����W"
	 ������">*:sequential/densenet169/conv5_block23_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1344]"�  "�����W"
	 �܈���">*:sequential/densenet169/conv5_block23_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1344]"�  "���ʐW"
	 ������">*:sequential/densenet169/conv5_block23_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1344]"�  "�����W"
	 ������">*:sequential/densenet169/conv5_block23_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1344]"�  "CU�팒W���"��" *$$1"!*size:5376 dest:0 async:1"�  "CU��͖W��"��" *$$1"!*size:5376 dest:0 async:1"�  ",`����Wء�"��" *$$1""8�"�  ",W����Wఉ"��" *$$1""8�"�  "�賅�W"
	 ������"<*8sequential/densenet169/conv5_block23_0_bn/AssignNewValue"
 �������"  "�  "��գ�W"
	 ������">*:sequential/densenet169/conv5_block23_0_bn/AssignNewValue_1"
 �������"  "�  "�����W"
	 ����"6*2sequential/densenet169/conv5_block23_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block23_1_conv/Conv2D"
 �������"*temp" "*[128,1344,1,1]"�  ",R��ŭW��"��" *$$1""8�"�  ",aБ�W���"��" *$$1""8�"�  "�ؒ��W"
	 ������"6*2sequential/densenet169/conv5_block23_1_conv/Conv2D"
 �������"  "�  "�����W"
	 ������">*:sequential/densenet169/conv5_block23_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �܉���">*:sequential/densenet169/conv5_block23_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����W"
	 ������">*:sequential/densenet169/conv5_block23_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����W"
	 ������">*:sequential/densenet169/conv5_block23_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�в��W"
	 ������">*:sequential/densenet169/conv5_block23_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����W���"��" *$$1"!*size:512 dest:0 async:1"�  "BU���W���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����W��"��" *$$1""8�"�  ",W�ȣ�W���"��" *$$1""8�"�  "�赘�W"
	 ������"<*8sequential/densenet169/conv5_block23_1_bn/AssignNewValue"
 �������"  "�  "�����W"
	 ������">*:sequential/densenet169/conv5_block23_1_bn/AssignNewValue_1"
 �������"  "�  "�����W"
dv�?" ����" ��" ��"
	 ������"6*2sequential/densenet169/conv5_block23_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "���W"
	 ������"6*2sequential/densenet169/conv5_block23_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R�ê�W���"��" *$$1""8�"�  ",a���W���"��" *$$1""8�"�  "��݄�W"
dv�?" ����" ��	" ��	"
	 ������"6*2sequential/densenet169/conv5_block23_2_conv/Conv2D"
 �������"  "�  "�����W"
	 ������"6*2sequential/densenet169/conv5_block23_concat/concat"
 �������"
*output" "*[200,1376,1,1]"�  ",N����W��"��" *$$1""8�"�  ",N����W���"��" *$$1""8�"�  "w����W"
	 ������"
  "  "�  "�����W"
	 �Ȃ���">*:sequential/densenet169/conv5_block24_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1376,1,1]"�  "�����W"
	 ������">*:sequential/densenet169/conv5_block24_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1376]"�  "��Ĳ�W"
	 ������">*:sequential/densenet169/conv5_block24_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1376]"�  "����W"
	 ������">*:sequential/densenet169/conv5_block24_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1376]"�  "��ʆ�X"
	 �஦��">*:sequential/densenet169/conv5_block24_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1376]"�  "CU���X��"��" *$$1"!*size:5504 dest:0 async:1"�  "CU����X�ۍ"��" *$$1"!*size:5504 dest:0 async:1"�  ",`����X蕟"��" *$$1""8�"�  ",W���X�À"��" *$$1""8�"�  "����X"
	 ������"<*8sequential/densenet169/conv5_block24_0_bn/AssignNewValue"
 �������"  "�  "��偗X"
	 �ʋ���">*:sequential/densenet169/conv5_block24_0_bn/AssignNewValue_1"
 �������"  "�  "�����X"
	 ��Ũ��"6*2sequential/densenet169/conv5_block24_1_conv/Conv2D"
 �������"
*output" "*
	 ��̨��"6*2sequential/densenet169/conv5_block24_1_conv/Conv2D"
 �������"*temp" "*[128,1376,1,1]"�  ",R����X���
"��" *$$1""8�"�  ",a��ƭX���"��" *$$1""8�"�  "���̳X"
	 ��̨��"6*2sequential/densenet169/conv5_block24_1_conv/Conv2D"
 �������"  "�  "�؟��X"
	 ��̨��">*:sequential/densenet169/conv5_block24_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block24_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����X"
	 ������">*:sequential/densenet169/conv5_block24_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����X"
	 ������">*:sequential/densenet169/conv5_block24_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�迿�X"
	 ������">*:sequential/densenet169/conv5_block24_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�׹X��"��" *$$1"!*size:512 dest:0 async:1"�  "BU����XЩ�"��" *$$1"!*size:512 dest:0 async:1"�  ",`�ُ�Xغ�"��" *$$1""8�"�  ",W證�X�ޙ"��" *$$1""8�"�  "�ؽ��X"
	 �Ќ���"<*8sequential/densenet169/conv5_block24_1_bn/AssignNewValue"
 �������"  "�  "�����X"
	 �Ԍ���">*:sequential/densenet169/conv5_block24_1_bn/AssignNewValue_1"
 �������"  "�  "�����X"
	 ��Ҩ��"6*2sequential/densenet169/conv5_block24_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����X"
	 ��Ө��"6*2sequential/densenet169/conv5_block24_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����X���"��" *$$1""8�"�  ",a����X���"��" *$$1""8�"�  "��ܝ�X"
	 ��Ө��"6*2sequential/densenet169/conv5_block24_2_conv/Conv2D"
 �������"  "�  "�����X"
	 ��Ө��"6*2sequential/densenet169/conv5_block24_concat/concat"
 �������"
*output" "*[200,1408,1,1]"�  ",N���XГ�"��" *$$1""8�"�  ",N����X���"��" *$$1""8�"�  "w����X"
	 ��Ҩ��"
  "  "�  "�����X"
	 �Ș���">*:sequential/densenet169/conv5_block25_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1408,1,1]"�  "�����X"
	 ������">*:sequential/densenet169/conv5_block25_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1408]"�  "�л��X"
	 ������">*:sequential/densenet169/conv5_block25_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1408]"�  "����X"
	 �ʋ���">*:sequential/densenet169/conv5_block25_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1408]"�  "�����X"
	 ��Ҩ��">*:sequential/densenet169/conv5_block25_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1408]"�  "CU����X���"��" *$$1"!*size:5632 dest:0 async:1"�  "CU����X���"��" *$$1"!*size:5632 dest:0 async:1"�  ",`����Y�ʷ"��" *$$1""8�"�  ",W��ݎY���"��" *$$1""8�"�  "��ﲓY"
	 ������"<*8sequential/densenet169/conv5_block25_0_bn/AssignNewValue"
 �������"  "�  "����Y"
	 ������">*:sequential/densenet169/conv5_block25_0_bn/AssignNewValue_1"
 �������"  "�  "���ݘY"
	 ��ݩ��"6*2sequential/densenet169/conv5_block25_1_conv/Conv2D"
 �������"
*output" "*
	 �����"6*2sequential/densenet169/conv5_block25_1_conv/Conv2D"
 �������"*temp" "*[128,1408,1,1]"�  ",R����Y���"��" *$$1""8�"�  ",a��̣Y���"��" *$$1""8�"�  "��뎩Y"
	 �����"6*2sequential/densenet169/conv5_block25_1_conv/Conv2D"
 �������"  "�  "�����Y"
	 �����">*:sequential/densenet169/conv5_block25_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �Ќ���">*:sequential/densenet169/conv5_block25_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ܹ�Y"
	 �Ԍ���">*:sequential/densenet169/conv5_block25_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���Y"
	 ������">*:sequential/densenet169/conv5_block25_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����Y"
	 ������">*:sequential/densenet169/conv5_block25_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����Y���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����Y�Ӎ"��" *$$1"!*size:512 dest:0 async:1"�  ",`����Y���"��" *$$1""8�"�  ",W��üY���"��" *$$1""8�"�  "����Y"
	 �䌒��"<*8sequential/densenet169/conv5_block25_1_bn/AssignNewValue"
 �������"  "�  "�����Y"
	 �茒��">*:sequential/densenet169/conv5_block25_1_bn/AssignNewValue_1"
 �������"  "�  "�����Y"
	 �����"6*2sequential/densenet169/conv5_block25_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����Y"
	 �����"6*2sequential/densenet169/conv5_block25_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����Y���"��" *$$1""8�"�  ",a����Y���"��" *$$1""8�"�  "�����Y"
	 �����"6*2sequential/densenet169/conv5_block25_2_conv/Conv2D"
 �������"  "�  "�����Y"
	 �����"6*2sequential/densenet169/conv5_block25_concat/concat"
 �������"
*output" "*[200,1440,1,1]"�  ",N����Y���"��" *$$1""8�"�  ",Nо��Y��"��" *$$1""8�"�  "w����Y"
	 �����"
  "  "�  "�����Y"
	 �ر���">*:sequential/densenet169/conv5_block26_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1440,1,1]"�  "��֙�Y"
	 ������">*:sequential/densenet169/conv5_block26_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1440]"�  "�����Y"
	 ��Ҩ��">*:sequential/densenet169/conv5_block26_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1440]"�  "�����Y"
	 ��Ҩ��">*:sequential/densenet169/conv5_block26_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1440]"�  "�����Y"
	 ��Ө��">*:sequential/densenet169/conv5_block26_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1440]"�  "CU����Y���"��" *$$1"!*size:5760 dest:0 async:1"�  "CU����Y��"��" *$$1"!*size:5760 dest:0 async:1"�  ",`����Y���"��" *$$1""8�"�  ",W����Y��"��" *$$1""8�"�  "�����Y"
	 ������"<*8sequential/densenet169/conv5_block26_0_bn/AssignNewValue"
 �������"  "�  "�����Y"
	 ������">*:sequential/densenet169/conv5_block26_0_bn/AssignNewValue_1"
 �������"  "�  "��އ�Y"
	 ������"6*2sequential/densenet169/conv5_block26_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block26_1_conv/Conv2D"
 �������"*temp" "*[128,1440,1,1]"�  ",R�ϲ�Y���"��" *$$1""8�"�  ",a����Z�я"��" *$$1""8�"�  "�褷�Z"
	 ������"6*2sequential/densenet169/conv5_block26_1_conv/Conv2D"
 �������"  "�  "�����Z"
	 ������">*:sequential/densenet169/conv5_block26_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �䌒��">*:sequential/densenet169/conv5_block26_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���˒Z"
	 �茒��">*:sequential/densenet169/conv5_block26_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�𫉓Z"
	 ������">*:sequential/densenet169/conv5_block26_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����Z"
	 ������">*:sequential/densenet169/conv5_block26_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����Z���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����Z���"��" *$$1"!*size:512 dest:0 async:1"�  ",`ؘ�Zع�"��" *$$1""8�"�  ",W��ǨZ��"��" *$$1""8�"�  "����Z"
	 ������"<*8sequential/densenet169/conv5_block26_1_bn/AssignNewValue"
 �������"  "�  "�����Z"
	 ������">*:sequential/densenet169/conv5_block26_1_bn/AssignNewValue_1"
 �������"  "�  "�����Z"
	 �����"6*2sequential/densenet169/conv5_block26_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "���ݱZ"
	 ������"6*2sequential/densenet169/conv5_block26_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R��۲Zز�"��" *$$1""8�"�  ",a���Z���"��" *$$1""8�"�  "�����Z"
	 ������"6*2sequential/densenet169/conv5_block26_2_conv/Conv2D"
 �������"  "�  "�����Z"
	 ������"6*2sequential/densenet169/conv5_block26_concat/concat"
 �������"
*output" "*[200,1472,1,1]"�  ",NЩ��Z���"��" *$$1""8�"�  ",N����Z���"��" *$$1""8�"�  "w藫�Z"
	 �����"
  "  "�  "�����Z"
	 ��̫��">*:sequential/densenet169/conv5_block27_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1472,1,1]"�  "�����Z"
	 ������">*:sequential/densenet169/conv5_block27_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1472]"�  "����Z"
	 ������">*:sequential/densenet169/conv5_block27_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1472]"�  "����Z"
	 �����">*:sequential/densenet169/conv5_block27_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1472]"�  "����Z"
	 �����">*:sequential/densenet169/conv5_block27_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1472]"�  "CU����Z���"��" *$$1"!*size:5888 dest:0 async:1"�  "CU����Z�ѓ"��" *$$1"!*size:5888 dest:0 async:1"�  ",`���Z�ڃ"��" *$$1""8�"�  ",W���Z���"��" *$$1""8�"�  "�����Z"
	 ������"<*8sequential/densenet169/conv5_block27_0_bn/AssignNewValue"
 �������"  "�  "�����Z"
	 ������">*:sequential/densenet169/conv5_block27_0_bn/AssignNewValue_1"
 �������"  "�  "�Ȼ��Z"
	 ������"6*2sequential/densenet169/conv5_block27_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block27_1_conv/Conv2D"
 �������"*temp" "*[128,1472,1,1]"�  ",R�ɵ�Z���"��" *$$1""8�"�  ",a��Z���"��" *$$1""8�"�  "�����Z"
	 ������"6*2sequential/densenet169/conv5_block27_1_conv/Conv2D"
 �������"  "�  "�����Z"
	 ������">*:sequential/densenet169/conv5_block27_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block27_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����Z"
	 ������">*:sequential/densenet169/conv5_block27_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����Z"
	 �Ʈ���">*:sequential/densenet169/conv5_block27_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����Z"
	 �ʮ���">*:sequential/densenet169/conv5_block27_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BUج��Z���"��" *$$1"!*size:512 dest:0 async:1"�  "BU���[��"��" *$$1"!*size:512 dest:0 async:1"�  ",`��܇[���"��" *$$1""8�"�  ",W��[���"��" *$$1""8�"�  "���ސ["
	 ������"<*8sequential/densenet169/conv5_block27_1_bn/AssignNewValue"
 �������"  "�  "�����["
	 ������">*:sequential/densenet169/conv5_block27_1_bn/AssignNewValue_1"
 �������"  "�  "�����["
	 �ࠬ��"6*2sequential/densenet169/conv5_block27_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����["
	 ������"6*2sequential/densenet169/conv5_block27_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����[Р�"��" *$$1""8�"�  ",a����[���"��" *$$1""8�"�  "���ˣ["
	 ������"6*2sequential/densenet169/conv5_block27_2_conv/Conv2D"
 �������"  "�  "�ؾ�["
	 ������"6*2sequential/densenet169/conv5_block27_concat/concat"
 �������"
*output" "*[200,1504,1,1]"�  ",N�ꂨ[���"��" *$$1""8�"�  ",N�ʉ�[���"��" *$$1""8�"�  "wؔ�["
	 �ࠬ��"
  "  "�  "���ְ["
	 �����">*:sequential/densenet169/conv5_block28_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1504,1,1]"�  "���["
	 ������">*:sequential/densenet169/conv5_block28_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1504]"�  "����["
	 �����">*:sequential/densenet169/conv5_block28_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1504]"�  "�����["
	 �����">*:sequential/densenet169/conv5_block28_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1504]"�  "�����["
Y�?" ����" �/" �0"
	 �ࠬ��">*:sequential/densenet169/conv5_block28_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1504]"�  "CU��ϳ[���"��" *$$1"!*size:6016 dest:0 async:1"�  "CU��ٷ[���"��" *$$1"!*size:6016 dest:0 async:1"�  ",`ಹ�[���"��" *$$1""8�"�  ",W����[�݇"��" *$$1""8�"�  "��щ�["
	 ������"<*8sequential/densenet169/conv5_block28_0_bn/AssignNewValue"
 �������"  "�  "�����["
	 �����">*:sequential/densenet169/conv5_block28_0_bn/AssignNewValue_1"
 �������"  "�  "�����["
	 ������"6*2sequential/densenet169/conv5_block28_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block28_1_conv/Conv2D"
 �������"*temp" "*[128,1504,1,1]"�  ",R���[�У"��" *$$1""8�"�  ",a����[���"��" *$$1""8�"�  "��Ӫ�["
	 ������"6*2sequential/densenet169/conv5_block28_1_conv/Conv2D"
 �������"  "�  "�����["
	 ������">*:sequential/densenet169/conv5_block28_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block28_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����["
	 ������">*:sequential/densenet169/conv5_block28_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����["
	 �ή���">*:sequential/densenet169/conv5_block28_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����["
	 �Ү���">*:sequential/densenet169/conv5_block28_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����[���"��" *$$1"!*size:512 dest:0 async:1"�  "BU��[Ѓ�"��" *$$1"!*size:512 dest:0 async:1"�  ",`����[���"��" *$$1""8�"�  ",W����[���"��" *$$1""8�"�  "�����["
	 ������"<*8sequential/densenet169/conv5_block28_1_bn/AssignNewValue"
 �������"  "�  "��޷�["
	 ������">*:sequential/densenet169/conv5_block28_1_bn/AssignNewValue_1"
 �������"  "�  "�����["
	 ������"6*2sequential/densenet169/conv5_block28_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����["
	 ��í��"6*2sequential/densenet169/conv5_block28_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",Rؠ��[�ަ"��" *$$1""8�"�  ",a����[��"��" *$$1""8�"�  "�����\"
	 ��í��"6*2sequential/densenet169/conv5_block28_2_conv/Conv2D"
 �������"  "�  "��륇\"
	 ��í��"6*2sequential/densenet169/conv5_block28_concat/concat"
 �������"
*output" "*[200,1536,1,1]"�  ",N����\���"��" *$$1""8�"�  ",N��͋\軿"��" *$$1""8�"�  "w��ݍ\"
	 ������"
  "  "�  "�����\"
	 ������">*:sequential/densenet169/conv5_block29_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1536,1,1]"�  "��狑\"
	 �����">*:sequential/densenet169/conv5_block29_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1536]"�  "�����\"
�R�?" ����" �0" �T"
	 ������">*:sequential/densenet169/conv5_block29_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1536]"�  "�����\"
	 ������">*:sequential/densenet169/conv5_block29_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1536]"�  "���ђ\"
	 ������">*:sequential/densenet169/conv5_block29_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1536]"�  "CU���\���"��" *$$1"!*size:6144 dest:0 async:1"�  "CU���\���"��" *$$1"!*size:6144 dest:0 async:1"�  ",`ؐ��\��"��" *$$1""8�"�  ",W�մ�\�Ŋ"��" *$$1""8�"�  "��ꔦ\"
�R�?" ����" �0" �X"
	 ������"<*8sequential/densenet169/conv5_block29_0_bn/AssignNewValue"
 �������"  "�  "��Ĩ�\"
	 �臝��">*:sequential/densenet169/conv5_block29_0_bn/AssignNewValue_1"
 �������"  "�  "�訣�\"
	 ��ٮ��"6*2sequential/densenet169/conv5_block29_1_conv/Conv2D"
 �������"
*output" "*
	 ��߮��"6*2sequential/densenet169/conv5_block29_1_conv/Conv2D"
 �������"*temp" "*[128,1536,1,1]"�  ",R����\���"��" *$$1""8�"�  ",a𧼴\���"��" *$$1""8�"�  "��ߴ�\"
	 ��߮��"6*2sequential/densenet169/conv5_block29_1_conv/Conv2D"
 �������"  "�  "����\"
	 ��߮��">*:sequential/densenet169/conv5_block29_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block29_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����\"
	 ������">*:sequential/densenet169/conv5_block29_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����\"
	 �֮���">*:sequential/densenet169/conv5_block29_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���о\"
	 �ڮ���">*:sequential/densenet169/conv5_block29_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU���\��"��" *$$1"!*size:512 dest:0 async:1"�  "BU����\���"��" *$$1"!*size:512 dest:0 async:1"�  ",`�֯�\���"��" *$$1""8�"�  ",W����\���"��" *$$1""8�"�  "�����\"
	 ������"<*8sequential/densenet169/conv5_block29_1_bn/AssignNewValue"
 �������"  "�  "�����\"
	 ������">*:sequential/densenet169/conv5_block29_1_bn/AssignNewValue_1"
 �������"  "�  "�،��\"
	 ������"6*2sequential/densenet169/conv5_block29_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����\"
	 �����"6*2sequential/densenet169/conv5_block29_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R����\���"��" *$$1""8�"�  ",a����\П�"��" *$$1""8�"�  "�����\"
	 �����"6*2sequential/densenet169/conv5_block29_2_conv/Conv2D"
 �������"  "�  "�প�\"
	 �����"6*2sequential/densenet169/conv5_block29_concat/concat"
 �������"
*output" "*[200,1568,1,1]"�  ",N����\���"��" *$$1""8�"�  ",N����\���"��" *$$1""8�"�  "wȘ��\"
	 ������"
  "  "�  "�����\"
	 ������">*:sequential/densenet169/conv5_block30_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1568,1,1]"�  "�����\"
	 ����">*:sequential/densenet169/conv5_block30_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1568]"�  "�И��\"
	 ������">*:sequential/densenet169/conv5_block30_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1568]"�  "�����\"
	 ������">*:sequential/densenet169/conv5_block30_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1568]"�  "�����\"
	 ��­��">*:sequential/densenet169/conv5_block30_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1568]"�  "CU�պ�\�"��" *$$1"!*size:6272 dest:0 async:1"�  "CU���\��"��" *$$1"!*size:6272 dest:0 async:1"�  ",`ؚ��\���"��" *$$1""8�"�  ",W����]���"��" *$$1""8�"�  "���̈]"
	 ������"<*8sequential/densenet169/conv5_block30_0_bn/AssignNewValue"
 �������"  "�  "����]"
	 ������">*:sequential/densenet169/conv5_block30_0_bn/AssignNewValue_1"
 �������"  "�  "����]"
	 ������"6*2sequential/densenet169/conv5_block30_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block30_1_conv/Conv2D"
 �������"*temp" "*[128,1568,1,1]"�  ",R𙀏]���"��" *$$1""8�"�  ",a����]���"��" *$$1""8�"�  "���՜]"
	 ������"6*2sequential/densenet169/conv5_block30_1_conv/Conv2D"
 �������"  "�  "�����]"
	 ������">*:sequential/densenet169/conv5_block30_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block30_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����]"
	 ������">*:sequential/densenet169/conv5_block30_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "����]"
	 �ޮ���">*:sequential/densenet169/conv5_block30_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����]"
	 �⮑��">*:sequential/densenet169/conv5_block30_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU�˜�]���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����]���"��" *$$1"!*size:512 dest:0 async:1"�  ",`���]���"��" *$$1""8�"�  ",W��ı]���"��" *$$1""8�"�  "���ն]"
	 ������"<*8sequential/densenet169/conv5_block30_1_bn/AssignNewValue"
 �������"  "�  "����]"
	 ������">*:sequential/densenet169/conv5_block30_1_bn/AssignNewValue_1"
 �������"  "�  "�����]"
	 ������"6*2sequential/densenet169/conv5_block30_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "���ڽ]"
	 ������"6*2sequential/densenet169/conv5_block30_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R��տ]��"��" *$$1""8�"�  ",a�ݥ�]���"��" *$$1""8�"�  "��߬�]"
	 ������"6*2sequential/densenet169/conv5_block30_2_conv/Conv2D"
 �������"  "�  "�����]"
	 ������"6*2sequential/densenet169/conv5_block30_concat/concat"
 �������"
*output" "*[200,1600,1,1]"�  ",Nȗ��]���"��" *$$1""8�"�  ",NȜ��]���"��" *$$1""8�"�  "wؖ��]"
	 ������"
  "  "�  "��İ�]"
	 ��۰��">*:sequential/densenet169/conv5_block31_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1600,1,1]"�  "��օ�]"
	 ��­��">*:sequential/densenet169/conv5_block31_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1600]"�  "�؎��]"
	 ��­��">*:sequential/densenet169/conv5_block31_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1600]"�  "�Ъ��]"
	 ������">*:sequential/densenet169/conv5_block31_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1600]"�  "�����]"
	 ������">*:sequential/densenet169/conv5_block31_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1600]"�  "CU�Ǹ�]���"��" *$$1"!*size:6400 dest:0 async:1"�  "CU����]��"��" *$$1"!*size:6400 dest:0 async:1"�  ",`����]ȗ�"��" *$$1""8�"�  ",W����]���"��" *$$1""8�"�  "�����]"
	 ��̟��"<*8sequential/densenet169/conv5_block31_0_bn/AssignNewValue"
 �������"  "�  "�ز��]"
	 ��̟��">*:sequential/densenet169/conv5_block31_0_bn/AssignNewValue_1"
 �������"  "�  "�����]"
	 ������"6*2sequential/densenet169/conv5_block31_1_conv/Conv2D"
 �������"
*output" "*
	 ������"6*2sequential/densenet169/conv5_block31_1_conv/Conv2D"
 �������"*temp" "*[128,1600,1,1]"�  ",R����]���"��" *$$1""8�"�  ",a����^ؓ�"��" *$$1""8�"�  "���ߊ^"
	 ������"6*2sequential/densenet169/conv5_block31_1_conv/Conv2D"
 �������"  "�  "�����^"
	 ������">*:sequential/densenet169/conv5_block31_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 �殑��">*:sequential/densenet169/conv5_block31_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����^"
	 ������">*:sequential/densenet169/conv5_block31_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "��ݢ�^"
	 ������">*:sequential/densenet169/conv5_block31_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "���֎^"
	 �ڄ���">*:sequential/densenet169/conv5_block31_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU��ُ^�"��" *$$1"!*size:512 dest:0 async:1"�  "BU�ϴ�^���"��" *$$1"!*size:512 dest:0 async:1"�  ",`����^���"��" *$$1""8�"�  ",W����^���"��" *$$1""8�"�  "��»�^"
	 ������"<*8sequential/densenet169/conv5_block31_1_bn/AssignNewValue"
 �������"  "�  "��ʋ�^"
	 ������">*:sequential/densenet169/conv5_block31_1_bn/AssignNewValue_1"
 �������"  "�  "�����^"
	 ������"6*2sequential/densenet169/conv5_block31_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�����^"
	 �ص���"6*2sequential/densenet169/conv5_block31_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R�ڠ�^ș�"��" *$$1""8�"�  ",a����^���"��" *$$1""8�"�  "����^"
	 �ص���"6*2sequential/densenet169/conv5_block31_2_conv/Conv2D"
 �������"  "�  "���^"
	 �ص���"6*2sequential/densenet169/conv5_block31_concat/concat"
 �������"
*output" "*[200,1632,1,1]"�  ",N�ƪ�^�Թ"��" *$$1""8�"�  ",N����^з�"��" *$$1""8�"�  "w����^"
	 ������"
  "  "�  "�����^"
	 ������">*:sequential/densenet169/conv5_block32_0_bn/FusedBatchNormV3"
 �������"
*output" "*[200,1632,1,1]"�  "��׾�^"
	 ��̟��">*:sequential/densenet169/conv5_block32_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1632]"�  "�����^"
	 ������">*:sequential/densenet169/conv5_block32_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1632]"�  "�����^"
	 �䋰��">*:sequential/densenet169/conv5_block32_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1632]"�  "����^"
	 ������">*:sequential/densenet169/conv5_block32_0_bn/FusedBatchNormV3"
 �������"
*output" "
*[1632]"�  "CUȗ��^���"��" *$$1"!*size:6528 dest:0 async:1"�  "CU����^���"��" *$$1"!*size:6528 dest:0 async:1"�  ",`����^���"��" *$$1""8�"�  ",W����^���"��" *$$1""8�"�  "�����^"
	 �ҕ���"<*8sequential/densenet169/conv5_block32_0_bn/AssignNewValue"
 �������"  "�  "�����^"
	 �����">*:sequential/densenet169/conv5_block32_0_bn/AssignNewValue_1"
 �������"  "�  "�����^"
	 ��ղ��"6*2sequential/densenet169/conv5_block32_1_conv/Conv2D"
 �������"
*output" "*
	 ��۲��"6*2sequential/densenet169/conv5_block32_1_conv/Conv2D"
 �������"*temp" "*[128,1632,1,1]"�  ",R����^���"��" *$$1""8�"�  ",a����^��"��" *$$1""8�"�  "����^"
	 ��۲��"6*2sequential/densenet169/conv5_block32_1_conv/Conv2D"
 �������"  "�  "�����^"
	 ��۲��">*:sequential/densenet169/conv5_block32_1_bn/FusedBatchNormV3"
 �������"
*output" "*
	 ������">*:sequential/densenet169/conv5_block32_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����^"
	 ������">*:sequential/densenet169/conv5_block32_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����^"
	 �ބ���">*:sequential/densenet169/conv5_block32_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "�����^"
	 �ℒ��">*:sequential/densenet169/conv5_block32_1_bn/FusedBatchNormV3"
 �������"
*output" "	*[128]"�  "BU����^���"��" *$$1"!*size:512 dest:0 async:1"�  "BU����^�"��" *$$1"!*size:512 dest:0 async:1"�  ",`����^�Ш"��" *$$1""8�"�  ",W�Ћ�_���"��" *$$1""8�"�  "����_"
	 �����"<*8sequential/densenet169/conv5_block32_1_bn/AssignNewValue"
 �������"  "�  "�����_"
	 ������">*:sequential/densenet169/conv5_block32_1_bn/AssignNewValue_1"
 �������"  "�  "�Єӈ_"
	 �����"6*2sequential/densenet169/conv5_block32_2_conv/Conv2D"
 �������"
*output" "*[200,32,1,1]"�  "�ȧ�_"
	 �����"6*2sequential/densenet169/conv5_block32_2_conv/Conv2D"
 �������"*temp" "*[32,128,3,3]"�  ",R��؊_�ͤ"��" *$$1""8�"�  ",a��Ӓ_���"��" *$$1""8�"�  "�Ⱥ��_"
	 �����"6*2sequential/densenet169/conv5_block32_2_conv/Conv2D"
 �������"  "�  "��׵�_"
	 �����"6*2sequential/densenet169/conv5_block32_concat/concat"
 �������"
*output" "*[200,1664,1,1]"�  ",N��М_��"��" *$$1""8�"�  ",N�֜�_���"��" *$$1""8�"�  "w����_"
	 �����"
  "  "�  "�����_"
	 ������".**sequential/densenet169/bn/FusedBatchNormV3"
 �������"
*output" "*[200,1664,1,1]"�  "���ͥ_"
	 �����".**sequential/densenet169/bn/FusedBatchNormV3"
 �������"
*output" "
*[1664]"�  "�����_"
	 �ҕ���".**sequential/densenet169/bn/FusedBatchNormV3"
 �������"
*output" "
*[1664]"�  "��˥�_"
	 �����".**sequential/densenet169/bn/FusedBatchNormV3"
 �������"
*output" "
*[1664]"�  "�����_"
	 �����".**sequential/densenet169/bn/FusedBatchNormV3"
 �������"
*output" "
*[1664]"�  "CUа��_Ș�"��" *$$1"!*size:6656 dest:0 async:1"�  "CU�ܝ�_���"��" *$$1"!*size:6656 dest:0 async:1"�  ",`���_ඇ"��" *$$1""8�"�  ",W���_���"��" *$$1""8�"�  "���ú_"
	 ������",*(sequential/densenet169/bn/AssignNewValue"
 �������"  "�  "���_"
	 �����".**sequential/densenet169/bn/AssignNewValue_1"
 �������"  "�  "�ؑ��_"*gpu_host_bfc"  " ��"	 �����"  " ��" " �"
	 ������"n*jgradient_tape/sequential/global_average_pooling2d/Shape-0-1-DataFormatVecPermuteNCHWToNHWC-LayoutOptimizer"
 �������"
*output" "*[4]"�  "�����_"
	 �����"*sequential/dense/MatMul"
 �������"
*output" "*[200,10]"�  ",b����_��"��" *$$1""8�"�  ",c����_ȕ�"��" *$$1""8�"�  "�����_"
	 �����"*SameWorkerRecvDone"
  "*dynamic" "*[4]"�  "A4����_Щ�"��" *$$1"!*size:16 dest:0 async:1"�  "�����_"*gpu_host_bfc"  " ��"	 �����"  " ��" " �"
	 ������">*:gradient_tape/sequential/global_average_pooling2d/floordiv"
 �������"
*output" "*[]"�  "�����_"
	 �愒��">*:categorical_crossentropy/softmax_cross_entropy_with_logits"
 �������"*temp" "*[200,1]"�  "�����_"
	 ���">*:categorical_crossentropy/softmax_cross_entropy_with_logits"
 �������"
*output" "	*[200]"�  "����_"
	 ������">*:categorical_crossentropy/softmax_cross_entropy_with_logits"
 �������"
*output" "*[200,10]"�  ",d����_���"��" *$$1""8�"�  ",e���`���"��" *$$1""8�"�  ",f��҇`���"��" *$$1""8�"�  "����`"
	 �Ѕ���">*:categorical_crossentropy/softmax_cross_entropy_with_logits"
 �������"  "�  "��٪�`"
"
	 ������">*:categorical_crossentropy/softmax_cross_entropy_with_logits"
 �������"  "�  ",gȯ��`輒"��" *$$1""8�"�  ",hЛڎ`�͂"��" *$$1""8�"�  ",i���`���"��" *$$1""8�"�  ",jЛ��`�͊"��" *$$1""8�"�  "��艝`"
	 �愒��">*:categorical_crossentropy/softmax_cross_entropy_with_logits"
 �������"  "�  "u�ԗ�`"
	 �����"
  "  "�  "�����`"
	 �����"*sequential/dense/Softmax"
 �������"*temp" "*[200,10]"�  "���ʡ`"
	 ������"*sequential/dense/Softmax"
 �������"*temp" "*[200,10]"�  ",k�ɢ`إ�"��" *$$1""8�"�  ",l讉�`���"��" *$$1""8�"�  ",m��ɨ`���"��" *$$1""8�"�  "�����`"
	 ������"*sequential/dense/Softmax"
 �������"  "�  "�����`"
	 �����"*sequential/dense/Softmax"
 �������"  "�  "�����`"
	 �����".**categorical_crossentropy/weighted_loss/Sum"
 �������"*temp" "*[]"�  ",n����`��"��" *$$1""8�"�  "�����`"*gpu_host_bfc"  " ��"	 �����"  " ��" " �"
	 ������"7*3categorical_crossentropy/weighted_loss/num_elements"
 �������"
*output" "*[]"�  "u�Ū�`"
	 ���"
  "  "�  "�����`"
	 �愒��"*ArgMax_1"
 �������"
*output" 	"	*[200]"�  ",o��ٸ`ȁ�"��" *$$1""8�"�  "uв�`"
	 �����"
  "  "�  "���߾`"
	 ������"*SameWorkerRecvDone"
  "*dynamic" "*[]"�  "@4���`ȋ�"��" *$$1"!*size:4 dest:0 async:1"�  "�����`"
	 ������"	*Equal"
 �������"
*output" 
"	*[200]"�  ",p����`���"��" *$$1""8�"�  "u����`"
	 ������"
  "  "�  "u����`"
	 �愒��"
  "  "�  "�����`"
	 ������"*SameWorkerRecvDone"
  "*dynamic" "*[]"�  "@4����`���"��" *$$1"!*size:4 dest:0 async:1"�  "����`"
	 ������"
*Cast_1"
 �������"
*output" "	*[200]"�  ",q����`���"��" *$$1""8�"�  "u�ũ�`"
	 ������"
  "  "�  "�����`"*gpu_host_bfc"  " ��"	 �����"  " ��" " �"
	 ������"*Size"
 �������"
*output" "*[]"�  "�����`"
	 ������"	*Sum_2"
 �������"*temp" "*[]"�  ",n����`��"��" *$$1""8�"�  "u����`"
	 ������"
  "  "�  ",r����`���"��" *$$1""8�"�  "t����`"
	 ������"
  "  "�  "�����`"
	 ������"*SameWorkerRecvDone"
  "*dynamic" "*[]"�  "@4���`���"��" *$$1"!*size:4 dest:0 async:1"�  ">����l���$"
 �������"  " "
	 ������"C*?gradient_tape/sequential/global_average_pooling2d/DynamicStitch"
 �������"
*output" "*[4]"�  ",s����m���"��" *$$1""8�"�  "t���m"
	 �����"
  "  "�  "t��ŏm"
	 �ژ���"
  "  "�  "��ҙ�m"*gpu_host_bfc"  " ��"	 �����"  " ��" " �"
	 ������"*SameWorkerRecvDone"
  "*dynamic" "*[4]"�  "A9Э��m���"��" *$$1"!*size:16 dest:0 async:1"�  "?����n����}"
 �������"  " "
	 ������"<*8categorical_crossentropy/weighted_loss/num_elements/Cast"
 �������"
*output" "*[]"�  ",t����n���"��" *$$1""8�"�  "t����n"
	 ������"
  "  "�  ",u�ʳ�nȄ�"��" *$$1""8�"�  ",u����n���	"��" *$$1""8�"�  ",8��oؙ�"��" *$$1""8�"�  ",r����o���"��" *$$1""8�"�  "t�梕o"
	 �����"
  "  "�  "��Լ�o"
	 �����"?*;gradient_tape/categorical_crossentropy/weighted_loss/Tile_1"
 �������"
*output" "	*[200]"�  ",v��șo�ɚ"��" *$$1""8�"�  "t���o"
	 ������"
  "  "�  ",r�ї�o���"��" *$$1""8�"�  "t����o"
	 ����"
  "  "�  ",?����o���"��" *$$1""8�"�  "u����o"
	 �����"
  "  "�  "�����o"
	 ����"*
div_no_nan"
 �������"
*output" "*[]"�  ",u����o��"��" *$$1""8�"�  "�����o"
	 �����"6*2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad"
 �������"
*output" "*[10]"�  ",w��¹o���"��" *$$1""8�"�  "��νo"
	 �Ѕ���")*%gradient_tape/sequential/dense/MatMul"
 �������"
*output" "*
[200,1664]"�  ",x���o���"��" *$$1""8�"�  "�����o"
	 ��ִ��"+*'gradient_tape/sequential/dense/MatMul_1"
 �������"
*output" "
	 ������"
  "  "�  ",z����o���"��" *$$1""8�"�  "t����o"
	 �����"
  "  "�  "m����o"*gpu_host_bfc"  " ��"	 �����"  " ��" " �"
	 ������"
  "  "�  ",z蘋�oȅ�"��" *$$1""8�"�  "wУ��o"
	 ��ִ��"
  "  "�  "mм��o"*gpu_host_bfc"  " ��"	 �����"  " ��" " �"
	 ������"
  "  "�  ",{���o��"��" *$$1""8�"�  "t����o"
	 �ژ���"
  "  "�  ",|�؀�o���"��" *$$1""8�"�  "w����o"
	 ������"
  "  "�  "�����o"
	 ������"@*<gradient_tape/sequential/densenet169/bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1664,1,1]"�  "�����o"
	 �����"@*<gradient_tape/sequential/densenet169/bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1664]"�  "�����o"
	 �����"@*<gradient_tape/sequential/densenet169/bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1664]"�  ",}����o���"��" *$$1""8�"�  "w���o"
	 �Ѕ���"
  "  "�  "w����o"
	 �����"
  "  "�  "u���o"
	 �����"
  "  "�  "u����o"
	 �����"
  "  "�  "�����o"
	 �����"E*Agradient_tape/sequential/densenet169/conv5_block32_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  ",~Ȧ��o���"��" *$$1""8�"�  "���ƃp"
	 �����"C*?gradient_tape/sequential/densenet169/conv5_block32_concat/Slice"
 �������"
*output" "*[200,1632,1,1]"�  ",~��Ȅp�Ղ"��" *$$1""8�	"�  "w�Й�p"
	 ������"
  "  "�  ",z��Ɗp��"��" *$$1""8�	"�  "u�䷍p"
	 �����"
  "  "�  ",z�တp���"��" *$$1""8�	"�  "u����p"
	 �����"
  "  "�  "�����p"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block32_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "���ܗp"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block32_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�����p"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block32_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  ",�ш�p���"��" *$$1""8�	"�  ",R���p�؎"��" *$$1""8�	"�  "���ܱp"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block32_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���òp"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block32_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����p"
ܺ��:�?" ����" ��" ��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ��ó��"X*Tgradient_tape/sequential/densenet169/conv5_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  ",RЉϷp��"��" *$$1""8�	"�  "����p"
	 ��̳��"X*Tgradient_tape/sequential/densenet169/conv5_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �愒��"X*Tgradient_tape/sequential/densenet169/conv5_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S����p��"��" *$$1""8�	"�  "-�����p���"��" *$$1""8�	"�  "-�����p���"��" *$$1""8�	"�  "�����p"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����p"
	 �愒��"X*Tgradient_tape/sequential/densenet169/conv5_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����p"
ܺ��:�?" ����" ��	" ��	"
	 ��ó��"X*Tgradient_tape/sequential/densenet169/conv5_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "wȭ��p"
	 �����"
  "  "�  ",z����p�λ"��" *$$1""8�	"�  "wȲ��p"
	 ������"
  "  "�  ",|���p���"��" *$$1""8�	"�  "w����p"
	 ��۲��"
  "  "�  "�����p"
	 ��۲��"P*Lgradient_tape/sequential/densenet169/conv5_block32_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block32_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "��Ɗ�p"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block32_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  ",}����p���"��" *$$1""8�	"�  "w����p"
	 ��̳��"
  "  "�  "w�̬�p"
	 ��ղ��"
  "  "�  "u����p"
	 �ބ���"
  "  "�  "u����p"
	 �ℒ��"
  "  "�  "����p"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block32_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1632,128]"�  "�س��p"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block32_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1632,1,1]"�  "�����p"
1�Tm7�?" ����" �" �""
	 �ބ���"Y*Ugradient_tape/sequential/densenet169/conv5_block32_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2292]"�  "-�Ѝ��pȭ�"��" *$$1""8�	"�  "-�����p���"��" *$$1""8�	"�  "-���q���"��" *$$1""8�	"�  "-����qЧ�"��" *$$1""8�	"�  ",R����q���"��" *$$1""8�	"�  "�����q"
	 �ބ���"Y*Ugradient_tape/sequential/densenet169/conv5_block32_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����q"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block32_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ū�q"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1632,1,1]"�  "����q"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1632,1,1]"�  ",R���q���"��" *$$1""8�	"�  "�����q"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1632,1,1]"�  "�����q"
	 �ބ���"X*Tgradient_tape/sequential/densenet169/conv5_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",Sجțqظ�"��" *$$1""8�	"�  "-�����q���"��" *$$1""8�	"�  "-��¹�q���"��" *$$1""8�	"�  "��ެ�q"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����q"
	 �ބ���"X*Tgradient_tape/sequential/densenet169/conv5_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����q"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w��˦q"
	 ��۲��"
  "  "�  ",z�ּ�q���"��" *$$1""8�	"�  "u����q"
	 �����"
  "  "�  ",z����q���"��" *$$1""8�	"�  "uȴ��q"
	 ������"
  "  "�  ",z����q���"��" *$$1""8�	"�  "w���q"
	 ������"
  "  "�  ",|����q�Ѡ"��" *$$1""8�	"�  "wЉ��q"
	 ������"
  "  "�  "��׾�q"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block32_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1632,1,1]"�  "�����q"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block32_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1632]"�  "���ֿq"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block32_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1632]"�  ",}����q���"��" *$$1""8�	"�  "w�Ӎ�q"
	 �����"
  "  "�  "w����q"
	 �ص���"
  "  "�  "u����q"
	 �䋰��"
  "  "�  "u����q"
	 ������"
  "  "�  ",zЧ��q���"��" *$$1""8�	"�  "u���q"
	 �����"
  "  "�  ",z����q���"��" *$$1""8�	"�  "u����q"
	 ������"
  "  "�  "-�����q���"��" *$$1""8�	"�  "w�Հ�q"
	 �����"
  "  "�  "�෻�q"
	 �ص���"E*Agradient_tape/sequential/densenet169/conv5_block31_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  ",~����qР�"��" *$$1""8�	"�  "����q"
	 ������"C*?gradient_tape/sequential/densenet169/conv5_block31_concat/Slice"
 �������"
*output" "*[200,1600,1,1]"�  ",~���q���"��" *$$1""8�	"�  "w荻�q"
	 ������"
  "  "�  "�����q"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block31_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�����q"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block31_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�����q"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block31_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  ",����q�ڊ"��" *$$1""8�	"�  ",R��q���"��" *$$1""8�	"�  "�����q"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block31_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����q"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block31_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����q"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �Д���"X*Tgradient_tape/sequential/densenet169/conv5_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  ",R�ǿ�q�Ϟ"��" *$$1""8�	"�  "���ār"
	 �Н���"X*Tgradient_tape/sequential/densenet169/conv5_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �ބ���"X*Tgradient_tape/sequential/densenet169/conv5_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S𻚉r��"��" *$$1""8�	"�  "-�����r���"��" *$$1""8�	"�  "-��Ϧ�r���"��" *$$1""8�	"�  "�����r"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���ϓr"
	 �ބ���"X*Tgradient_tape/sequential/densenet169/conv5_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����r"
	 �Д���"X*Tgradient_tape/sequential/densenet169/conv5_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w��۔r"
	 �ص���"
  "  "�  ",z����r���"��" *$$1""8�	"�  "w����r"
	 ������"
  "  "�  ",|�斝r�˕"��" *$$1""8�	"�  "w����r"
	 ������"
  "  "�  "���r"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block31_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block31_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "��µ�r"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block31_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  ",}Ȁ�rؽ�"��" *$$1""8�	"�  "w���r"
	 �Н���"
  "  "�  "w��Īr"
	 ������"
  "  "�  "u��ߪr"
	 ������"
  "  "�  "u�څ�r"
	 �ڄ���"
  "  "�  "���̬r"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block31_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1600,128]"�  "��ڣ�r"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block31_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1600,1,1]"�  "��Ȫ�r"
	 �ڄ���"Y*Ugradient_tape/sequential/densenet169/conv5_block31_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2292]"�  "-����r��"��" *$$1""8�	"�  "-�����rؙ�"��" *$$1""8�	"�  "-����rн�"��" *$$1""8�	"�  "-�����rЙ�"��" *$$1""8�	"�  ",R����r���"��" *$$1""8�	"�  "�����r"
	 �ڄ���"Y*Ugradient_tape/sequential/densenet169/conv5_block31_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����r"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block31_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����r"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1600,1,1]"�  "�����r"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1600,1,1]"�  ",R�¯�r���"��" *$$1""8�	"�  "�����r"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1600,1,1]"�  "�����r"
	 �ڄ���"X*Tgradient_tape/sequential/densenet169/conv5_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S���r���"��" *$$1""8�	"�  "-����r���"��" *$$1""8�	"�  "-�����r���"��" *$$1""8�	"�  "����r"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�踆�r"
	 �ڄ���"X*Tgradient_tape/sequential/densenet169/conv5_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��զ�r"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w�ֆ�r"
	 ������"
  "  "�  ",z����rЂ�"��" *$$1""8�	"�  "u����r"
	 �����"
  "  "�  ",z����r�͏"��" *$$1""8�	"�  "u����r"
	 ������"
  "  "�  ",z����r��"��" *$$1""8�	"�  "w����r"
	 ������"
  "  "�  ",|����r���"��" *$$1""8�	"�  "w���r"
	 ��۰��"
  "  "�  "��˷�r"
	 ��۰��"P*Lgradient_tape/sequential/densenet169/conv5_block31_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1600,1,1]"�  "��̊�r"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block31_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1600]"�  "����r"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block31_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1600]"�  ",}����r���"��" *$$1""8�	"�  "w����s"
	 ������"
  "  "�  "w����s"
	 ������"
  "  "�  "u���s"
	 ������"
  "  "�  "u�؟�s"
	 ������"
  "  "�  ",z�ϵ�s���"��" *$$1""8�	"�  "u����s"
	 �����"
  "  "�  ",z����sȄ�"��" *$$1""8�	"�  "u��Ñs"
	 ������"
  "  "�  "-��ț�s���"��" *$$1""8�	"�  "w����s"
	 ������"
  "  "�  "�����s"
	 �䋰��"E*Agradient_tape/sequential/densenet169/conv5_block30_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  ",~��јs���"��" *$$1""8�	"�  "��ﯜs"
	 ������"C*?gradient_tape/sequential/densenet169/conv5_block30_concat/Slice"
 �������"
*output" "*[200,1568,1,1]"�  ",~艬�s���"��" *$$1""8�	"�  "w���s"
	 ��۰��"
  "  "�  "�Ц��s"
	 ��۰��"Y*Ugradient_tape/sequential/densenet169/conv5_block30_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�����s"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block30_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "���Цs"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block30_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  ",��ˬs�ڗ"��" *$$1""8�	"�  ",R����s���"��" *$$1""8�	"�  "�𙿵s"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block30_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����s"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block30_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��Բ�s"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  ",R��ߺs�"��" *$$1""8�	"�  "���ܽs"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �ڄ���"X*Tgradient_tape/sequential/densenet169/conv5_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S����s���"��" *$$1""8�	"�  "-�����s���"��" *$$1""8�	"�  "-�����s��"��" *$$1""8�	"�  "�����s"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����s"
	 �ڄ���"X*Tgradient_tape/sequential/densenet169/conv5_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�Ф��s"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w����s"
	 �䋰��"
  "  "�  ",z����s���"��" *$$1""8�	"�  "w����s"
	 ��۰��"
  "  "�  ",|����s���"��" *$$1""8�	"�  "w����s"
	 ������"
  "  "�  "�����s"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block30_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block30_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "�����s"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block30_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  ",}跡�s���"��" *$$1""8�	"�  "w�ї�s"
	 �����"
  "  "�  "w����s"
	 ������"
  "  "�  "u����s"
	 �ޮ���"
  "  "�  "uȞ��s"
	 �⮑��"
  "  "�  "�؍��s"
	 ��۰��"Y*Ugradient_tape/sequential/densenet169/conv5_block30_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1568,128]"�  "�����s"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block30_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1568,1,1]"�  "�����s"
	 �ڄ���"Y*Ugradient_tape/sequential/densenet169/conv5_block30_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2292]"�  "-�����s���"��" *$$1""8�	"�  "-�����s���"��" *$$1""8�	"�  "-��ַ�s���"��" *$$1""8�	"�  "-�����s���"��" *$$1""8�	"�  ",R����sȶ�"��" *$$1""8�	"�  "���ԁt"
	 �ڄ���"Y*Ugradient_tape/sequential/densenet169/conv5_block30_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����t"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block30_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����t"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1568,1,1]"�  "���t"
	 ��ر��"X*Tgradient_tape/sequential/densenet169/conv5_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1568,1,1]"�  ",R����t���"��" *$$1""8�	"�  "��᷊t"
	 �Љ���"X*Tgradient_tape/sequential/densenet169/conv5_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1568,1,1]"�  "�����t"
	 �ڄ���"X*Tgradient_tape/sequential/densenet169/conv5_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S��ёt���"��" *$$1""8�	"�  "-��؄�t���"��" *$$1""8�	"�  "-�����t���"��" *$$1""8�	"�  "��˫�t"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ȇ�t"
	 �ڄ���"X*Tgradient_tape/sequential/densenet169/conv5_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�膪�t"
	 ��ر��"X*Tgradient_tape/sequential/densenet169/conv5_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w�Ί�t"
	 ������"
  "  "�  ",z��ϟt���"��" *$$1""8�	"�  "u��ȧt"
	 ������"
  "  "�  ",z��۪t�ߣ"��" *$$1""8�	"�  "u���t"
	 �����"
  "  "�  ",z����t���"��" *$$1""8�	"�  "w����t"
	 ��۰��"
  "  "�  ",|���t���"��" *$$1""8�	"�  "w�ηt"
�<I�?" ����" ��L" ��L"
	 ������"
  "  "�  "�����t"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block30_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1568,1,1]"�  "���ʹt"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block30_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1568]"�  "�����t"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block30_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1568]"�  ",}��t���"��" *$$1""8�	"�  "w�Ѿ�t"
	 �Љ���"
  "  "�  "w����t"
	 �����"
  "  "�  "uȲ��t"
	 ������"
  "  "�  "u����t"
	 ��­��"
  "  "�  ",zȨ��t���"��" *$$1""8�	"�  "uأ��t"
	 �����"
  "  "�  ",z����t���"��" *$$1""8�	"�  "u����t"
	 ������"
  "  "�  "-�����t࿄"��" *$$1""8�	"�  "w�ҝ�t"
	 ������"
  "  "�  "�����t"
	 �����"E*Agradient_tape/sequential/densenet169/conv5_block29_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  ",~����t���"��" *$$1""8�	"�  "�����t"
	 �����"C*?gradient_tape/sequential/densenet169/conv5_block29_concat/Slice"
 �������"
*output" "*[200,1536,1,1]"�  ",~ȥ��tح�"��" *$$1""8�	"�  "w����t"
	 ������"
  "  "�  "�����t"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block29_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�����t"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block29_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�����t"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block29_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  ",����t�Č"��" *$$1""8�	"�  ",R�˞�t���"��" *$$1""8�	"�  "�����t"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block29_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����t"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block29_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����t"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  ",R����t��"��" *$$1""8�	"�  "�����t"
	 ��ʯ��"X*Tgradient_tape/sequential/densenet169/conv5_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �ڄ���"X*Tgradient_tape/sequential/densenet169/conv5_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S����u���"��" *$$1""8�	"�  "-����u���"��" *$$1""8�	"�  "-�����u��"��" *$$1""8�	"�  "�Цωu"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�蜣�u"
	 �ڄ���"X*Tgradient_tape/sequential/densenet169/conv5_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���Ɗu"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w�է�u"
	 �����"
  "  "�  ",z����u���"��" *$$1""8�	"�  "w����u"
	 ������"
  "  "�  ",|���u��"��" *$$1""8�	"�  "wв��u"
	 ��߮��"
  "  "�  "�����u"
	 ��߮��"P*Lgradient_tape/sequential/densenet169/conv5_block29_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block29_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "�����u"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block29_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  ",}�ϭ�uȟ�"��" *$$1""8�	"�  "w����u"
	 ��ʯ��"
  "  "�  "w��ˠu"
	 ��ٮ��"
  "  "�  "u詭�u"
	 �֮���"
  "  "�  "u��סu"
	 �ڮ���"
  "  "�  "��Ձ�u"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block29_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1536,128]"�  "�،�u"
	 �䋰��"Y*Ugradient_tape/sequential/densenet169/conv5_block29_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1536,1,1]"�  "�؉ܨu"
	 �ڄ���"Y*Ugradient_tape/sequential/densenet169/conv5_block29_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2284]"�  "-�����u���"��" *$$1""8�	"�  "-�еϱu���"��" *$$1""8�	"�  "-�����u���"��" *$$1""8�	"�  "-�����u���"��" *$$1""8�	"�  ",R��ӹu���"��" *$$1""8�	"�  "�����u"
	 �ڄ���"Y*Ugradient_tape/sequential/densenet169/conv5_block29_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����u"
	 �䋰��"Y*Ugradient_tape/sequential/densenet169/conv5_block29_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����u"
	 �䋰��"X*Tgradient_tape/sequential/densenet169/conv5_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1536,1,1]"�  "�����u"
	 ��ְ��"X*Tgradient_tape/sequential/densenet169/conv5_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1536,1,1]"�  ",R����u�ơ"��" *$$1""8�	"�  "�����u"
	 �䆱��"X*Tgradient_tape/sequential/densenet169/conv5_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1536,1,1]"�  "�����u"
	 �ڄ���"X*Tgradient_tape/sequential/densenet169/conv5_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S�ɔ�u�"��" *$$1""8�	"�  "-��ݻ�u�ݹ"��" *$$1""8�	"�  "-�����u���"��" *$$1""8�	"�  "�����u"
	 �䋰��"X*Tgradient_tape/sequential/densenet169/conv5_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����u"
	 �ڄ���"X*Tgradient_tape/sequential/densenet169/conv5_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����u"
	 ��ְ��"X*Tgradient_tape/sequential/densenet169/conv5_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w����u"
	 ��߮��"
  "  "�  ",z����u���"��" *$$1""8�	"�  "u����u"
	 ������"
  "  "�  ",z����u���"��" *$$1""8�	"�  "u����u"
	 �����"
  "  "�  ",z�Ɛ�u���"��" *$$1""8�	"�  "w����u"
	 ������"
  "  "�  ",|����u���"��" *$$1""8�	"�  "w���u"
	 ������"
  "  "�  "�����u"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block29_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1536,1,1]"�  "�����u"
	 �臝��"P*Lgradient_tape/sequential/densenet169/conv5_block29_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1536]"�  "�����u"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block29_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1536]"�  ",}�܎�u���"��" *$$1""8�	"�  "w�у�u"
	 �䆱��"
  "  "�  "w����u"
	 ��í��"
  "  "�  "u����u"
	 ������"
  "  "�  "u����u"
	 ������"
  "  "�  ",z����u���"��" *$$1""8�	"�  "u����u"
	 �臝��"
  "  "�  ",z����u���"��" *$$1""8�	"�  "u諭�v"
	 �����"
  "  "�  "-����v���"��" *$$1""8�	"�  "w�菇v"
	 �����"
  "  "�  "��Կ�v"
	 ��í��"E*Agradient_tape/sequential/densenet169/conv5_block28_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  ",~���v���"��" *$$1""8�	"�  "�����v"
	 ��ĭ��"C*?gradient_tape/sequential/densenet169/conv5_block28_concat/Slice"
 �������"
*output" "*[200,1504,1,1]"�  ",~�㺎v���"��" *$$1""8�	"�  "w��ˑv"
	 ������"
  "  "�  "����v"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block28_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "���єv"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block28_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "���֗v"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block28_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  ",����v�׋"��" *$$1""8�	"�  ",R��٣v���"��" *$$1""8�	"�  "���v"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block28_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����v"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block28_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����v"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  ",R�۟�v���"��" *$$1""8�	"�  "��ڔ�v"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �ڄ���"X*Tgradient_tape/sequential/densenet169/conv5_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S���v���"��" *$$1""8�	"�  "-��v���"��" *$$1""8�	"�  "-���ʼv���"��" *$$1""8�	"�  "�����v"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���޿v"
	 �ڄ���"X*Tgradient_tape/sequential/densenet169/conv5_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����v"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w����v"
	 ��í��"
  "  "�  ",z����v���"��" *$$1""8�	"�  "wؾ��v"
	 ������"
  "  "�  ",|���v�Ǟ"��" *$$1""8�	"�  "w����v"
	 ������"
  "  "�  "�����v"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block28_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block28_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "�����v"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block28_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  ",}����v���"��" *$$1""8�	"�  "w����v"
	 ������"
  "  "�  "w���v"
	 ������"
  "  "�  "u����v"
	 �ή���"
  "  "�  "u���v"
	 �Ү���"
  "  "�  "��ʭ�v"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block28_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1504,128]"�  "�����v"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block28_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1504,1,1]"�  "�����v"
	 �ή���"Y*Ugradient_tape/sequential/densenet169/conv5_block28_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2284]"�  "-����v���"��" *$$1""8�	"�  "-����v���"��" *$$1""8�	"�  "-�����v���"��" *$$1""8�	"�  "-�����v���"��" *$$1""8�	"�  ",R�̑�v���"��" *$$1""8�
"�  "�����v"
	 �ή���"Y*Ugradient_tape/sequential/densenet169/conv5_block28_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����v"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block28_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����v"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1504,1,1]"�  "�����v"
	 �؆���"X*Tgradient_tape/sequential/densenet169/conv5_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1504,1,1]"�  ",RЉ��vȈ�"��" *$$1""8�
"�  "�Ѓ��v"
	 �ص���"X*Tgradient_tape/sequential/densenet169/conv5_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1504,1,1]"�  "�ؙ��v"
	 �ή���"X*Tgradient_tape/sequential/densenet169/conv5_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S���w�ڕ"��" *$$1""8�
"�  "-�����w��"��" *$$1""8�
"�  "-���ۆw���"��" *$$1""8�
"�  "��쫉w"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����w"
	 �ή���"X*Tgradient_tape/sequential/densenet169/conv5_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�𙟊w"
	 �؆���"X*Tgradient_tape/sequential/densenet169/conv5_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w�؂�w"
	 ������"
  "  "�  ",z����wР�"��" *$$1""8�
"�  "u�ع�w"
	 ������"
  "  "�  ",z�阔w�ӧ"��" *$$1""8�
"�  "u����w"
	 �����"
  "  "�  ",z����w���"��" *$$1""8�
"�  "w����w"
	 ������"
  "  "�  ",|����w��"��" *$$1""8�
"�  "wз��w"
	 �����"
  "  "�  "��ˬ�w"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block28_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1504,1,1]"�  "�����w"
	 �臝��"P*Lgradient_tape/sequential/densenet169/conv5_block28_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1504]"�  "��ࢣw"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block28_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1504]"�  ",}ț��w���"��" *$$1""8�
"�  "w����w"
	 �ص���"
  "  "�  "w��ʪw"
	 ������"
  "  "�  "u���w"
	 �����"
  "  "�  "u����w"
	 �ࠬ��"
  "  "�  ",z��ʮw衛"��" *$$1""8�
"�  "u��ñw"
��X�?" ����" �/" �0"
	 �臝��"
  "  "�  ",z����w���"��" *$$1""8�
"�  "u����w"
	 �����"
  "  "�  "-�����w���"��" *$$1""8�
"�  "w����w"
	 ��ĭ��"
  "  "�  "����w"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block27_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  ",~�̇�w��"��" *$$1""8�
"�  "�����w"
	 ����"C*?gradient_tape/sequential/densenet169/conv5_block27_concat/Slice"
 �������"
*output" "*[200,1472,1,1]"�  ",~����w���"��" *$$1""8�
"�  "w����w"
	 �����"
  "  "�  "�����w"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block27_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�����w"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block27_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "��٢�w"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block27_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  ",�͇�w���"��" *$$1""8�
"�  ",R����w���"��" *$$1""8�
"�  "�����w"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block27_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����w"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block27_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����w"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  ",R����w���"��" *$$1""8�
"�  "����w"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �ή���"X*Tgradient_tape/sequential/densenet169/conv5_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S�ɵ�w�Ҫ"��" *$$1""8�
"�  "-�����w�ѽ"��" *$$1""8�
"�  "-�����w���"��" *$$1""8�
"�  "�����w"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�軜�w"
	 �ή���"X*Tgradient_tape/sequential/densenet169/conv5_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�г��w"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w����w"
	 ������"
  "  "�  ",z����w���"��" *$$1""8�
"�  "w����w"
	 �����"
  "  "�  ",|����w�ԗ"��" *$$1""8�
"�  "w����w"
	 ������"
  "  "�  "�����w"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block27_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block27_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "�����x"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block27_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  ",}�ޙ�x��"��" *$$1""8�
"�  "w�ݕ�x"
	 ������"
  "  "�  "w��Èx"
	 ������"
  "  "�  "u���x"
	 �Ʈ���"
  "  "�  "u����x"
	 �ʮ���"
  "  "�  "���ʊx"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block27_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1472,128]"�  "�𶰌x"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv5_block27_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1472,1,1]"�  "�����x"
	 �Ʈ���"Y*Ugradient_tape/sequential/densenet169/conv5_block27_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2284]"�  "-���ԕx��"��" *$$1""8�
"�  "-�𡐙x���"��" *$$1""8�
"�  "-�؞�x���"��" *$$1""8�
"�  "-���Ϟx��"��" *$$1""8�
"�  ",R����x���"��" *$$1""8�
"�  "��أ�x"
	 �Ʈ���"Y*Ugradient_tape/sequential/densenet169/conv5_block27_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����x"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv5_block27_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��܉�x"
	 ��í��"X*Tgradient_tape/sequential/densenet169/conv5_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1472,1,1]"�  "���Ĩx"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1472,1,1]"�  ",R�湩x���"��" *$$1""8�
"�  "�����x"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1472,1,1]"�  "���˰x"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S襄�x���"��" *$$1""8�
"�  "-���Ÿx���"��" *$$1""8�
"�  "-�ء��x���"��" *$$1""8�
"�  "���Ƚx"
	 ��í��"X*Tgradient_tape/sequential/densenet169/conv5_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����x"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����x"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w����x"
	 ������"
  "  "�  ",z����x�ڢ"��" *$$1""8�
"�  "uȣ��x"
	 ������"
  "  "�  ",z����x���"��" *$$1""8�
"�  "u����x"
	 �����"
  "  "�  ",z����x��"��" *$$1""8�
"�  "w����x"
	 �����"
  "  "�  ",|����x���"��" *$$1""8�
"�  "w����x"
	 ��̫��"
  "  "�  "�����x"
	 ��̫��"P*Lgradient_tape/sequential/densenet169/conv5_block27_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1472,1,1]"�  "��Χ�x"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block27_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1472]"�  "�����x"
	 �臝��"P*Lgradient_tape/sequential/densenet169/conv5_block27_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1472]"�  ",}����x���"��" *$$1""8�
"�  "w����x"
	 ������"
  "  "�  "wО��x"
	 ������"
  "  "�  "u����x"
	 �����"
  "  "�  "u����x"
	 �����"
  "  "�  ",z�˶�x�Տ"��" *$$1""8�
"�  "u�ٞ�x"
	 ������"
  "  "�  ",z����x���"��" *$$1""8�
"�  "u�Ϛ�x"
	 �臝��"
  "  "�  "-�����xح�"��" *$$1""8�
"�  "w�ȭ�x"
	 ����"
  "  "�  "�����x"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block26_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  ",~૸�x�Ж"��" *$$1""8�
"�  "��ӏ�x"
	 ������"C*?gradient_tape/sequential/densenet169/conv5_block26_concat/Slice"
 �������"
*output" "*[200,1440,1,1]"�  ",~����x���"��" *$$1""8�
"�  "w����x"
	 ��̫��"
  "  "�  "�����x"
	 ��̫��"Y*Ugradient_tape/sequential/densenet169/conv5_block26_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�����x"
	 ��ի��"Y*Ugradient_tape/sequential/densenet169/conv5_block26_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�����x"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block26_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  ",�ɱ�y؃�"��" *$$1""8�
"�  ",R��ˊy���"��" *$$1""8�
"�  "�����y"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block26_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����y"
	 ��ի��"Y*Ugradient_tape/sequential/densenet169/conv5_block26_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���Ԑy"
	 ��ի��"X*Tgradient_tape/sequential/densenet169/conv5_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ��۫��"X*Tgradient_tape/sequential/densenet169/conv5_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  ",R����y���"��" *$$1""8�
"�  "�ȹĖy"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",Sز��y��"��" *$$1""8�
"�  "-�����y���"��" *$$1""8�
"�  "-�����y���"��" *$$1""8�
"�  "����y"
	 ��ի��"X*Tgradient_tape/sequential/densenet169/conv5_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���ʦy"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����y"
	 ��۫��"X*Tgradient_tape/sequential/densenet169/conv5_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w��ӧy"
	 ������"
  "  "�  ",z����y�ʪ"��" *$$1""8�
"�  "w����y"
	 ��̫��"
  "  "�  ",|����y�Ι"��" *$$1""8�
"�  "w�Έ�y"
	 ������"
  "  "�  "���y"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block26_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block26_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "��τ�y"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block26_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  ",}����y���"��" *$$1""8�
"�  "w��y"
	 �����"
  "  "�  "w�ӯ�y"
	 ������"
  "  "�  "u����y"
	 ������"
  "  "�  "u��Žy"
	 ������"
  "  "�  "����y"
	 ��̫��"Y*Ugradient_tape/sequential/densenet169/conv5_block26_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1440,128]"�  "����y"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block26_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1440,1,1]"�  "�����y"
	 �Ʈ���"Y*Ugradient_tape/sequential/densenet169/conv5_block26_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2284]"�  "-�����y���"��" *$$1""8�
"�  "-�����y���"��" *$$1""8�
"�  "-�����y���"��" *$$1""8�
"�  "-�����y�͏"��" *$$1""8�
"�  ",R����y���"��" *$$1""8�
"�  "�����y"
	 �Ʈ���"Y*Ugradient_tape/sequential/densenet169/conv5_block26_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��Ɠ�y"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block26_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����y"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1440,1,1]"�  "�����y"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1440,1,1]"�  ",R����y�"��" *$$1""8�
"�  "�����y"
	 ��í��"X*Tgradient_tape/sequential/densenet169/conv5_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1440,1,1]"�  "�غ��y"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S����y���"��" *$$1""8�
"�  "-��ҟ�y���"��" *$$1""8�
"�  "-����y���"��" *$$1""8�
"�  "��Ղ�y"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����y"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����y"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w����y"
	 ������"
  "  "�  ",z����y���"��" *$$1""8�
"�  "u����y"
	 ������"
  "  "�  ",zȷ��y���"��" *$$1""8�
"�  "u����y"
	 �����"
  "  "�  ",z����z���"��" *$$1""8�
"�  "w�殄z"
	 ��̫��"
  "  "�  ",|����z�ݔ"��" *$$1""8�
"�  "w���z"
	 �ر���"
  "  "�  "�м�z"
	 �ر���"P*Lgradient_tape/sequential/densenet169/conv5_block26_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1440,1,1]"�  "��׻�z"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block26_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1440]"�  "����z"
	 �臝��"P*Lgradient_tape/sequential/densenet169/conv5_block26_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1440]"�  ",}���z���"��" *$$1""8�
"�  "w����z"
~b�?" ����" ��F" ��F"
	 ��í��"
  "  "�  "w��ڒz"
	 �����"
  "  "�  "u����z"
	 ��Ҩ��"
  "  "�  "uࠠ�z"
	 ��Ө��"
  "  "�  ",z���z���"��" *$$1""8�
"�  "u���z"
	 ������"
  "  "�  ",z����z���"��" *$$1""8�
"�  "u����z"
	 �臝��"
  "  "�  "-�؄��zت�"��" *$$1""8�
"�  "wȐ��z"
	 ������"
  "  "�  "�����z"
	 �����"E*Agradient_tape/sequential/densenet169/conv5_block25_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  ",~����z���"��" *$$1""8�
"�  "��锩z"
	 �����"C*?gradient_tape/sequential/densenet169/conv5_block25_concat/Slice"
 �������"
*output" "*[200,1408,1,1]"�  ",~�풪z���"��" *$$1""8�
"�  "w����z"
	 �ر���"
  "  "�  "�ؤ��z"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block25_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�Ș��z"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block25_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "��ص�z"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block25_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  ",�圹zЉ�"��" *$$1""8�
"�  ",R��ȿz���"��" *$$1""8�
"�  "����z"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block25_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����z"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block25_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����z"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �ȱ���"X*Tgradient_tape/sequential/densenet169/conv5_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  ",R����z���"��" *$$1""8�
"�  "��Ͳ�z"
	 �Ⱥ���"X*Tgradient_tape/sequential/densenet169/conv5_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S����z�Ш"��" *$$1""8�
"�  "-�����z���"��" *$$1""8�
"�  "-�����z���"��" *$$1""8�
"�  "�����z"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��؋�z"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ӹ�z"
	 �ȱ���"X*Tgradient_tape/sequential/densenet169/conv5_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w�ؕ�z"
	 �����"
  "  "�  ",z����z���"��" *$$1""8�
"�  "w����z"
	 ������"
  "  "�  ",|����z�Õ"��" *$$1""8�
"�  "w����z"
	 �����"
  "  "�  "�����z"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block25_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block25_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "�����z"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block25_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  ",}����z���"��" *$$1""8�
"�  "w���z"
	 �Ⱥ���"
  "  "�  "w����z"
	 ��ݩ��"
  "  "�  "u����z"
	 ������"
  "  "�  "u�ϗ�z"
	 ������"
  "  "�  "�����z"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block25_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1408,128]"�  "�ذ��z"
	 ��ά��"Y*Ugradient_tape/sequential/densenet169/conv5_block25_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1408,1,1]"�  "��՚�z"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block25_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2276]"�  "-�����zȇ�"��" *$$1""8�
"�  "-�Ъ��{���"��" *$$1""8�
"�  "-���ބ{���"��" *$$1""8�
"�  "-�����{���"��" *$$1""8�
"�  ",R����{���"��" *$$1""8�
"�  "��ǜ�{"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block25_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��۝�{"
	 ��ά��"Y*Ugradient_tape/sequential/densenet169/conv5_block25_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��إ�{"
	 ��ά��"X*Tgradient_tape/sequential/densenet169/conv5_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1408,1,1]"�  "����{"
	 �ر���"X*Tgradient_tape/sequential/densenet169/conv5_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1408,1,1]"�  ",R���{Є�"��" *$$1""8�
"�  "�����{"
	 ��ݪ��"X*Tgradient_tape/sequential/densenet169/conv5_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1408,1,1]"�  "��׏�{"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S��ȝ{Р�"��" *$$1""8�
"�  "-����{ط�"��" *$$1""8�
"�  "-�����{���"��" *$$1""8�
"�  "���ϥ{"
	 ��ά��"X*Tgradient_tape/sequential/densenet169/conv5_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�Л��{"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����{"
	 �ر���"X*Tgradient_tape/sequential/densenet169/conv5_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w����{"
	 �����"
  "  "�  ",z��۪{���"��" *$$1""8�
"�  "u���{"
	 ������"
  "  "�  ",z�ݡ�{���"��" *$$1""8�
"�  "u���{"
	 �����"
  "  "�  ",z����{���"��" *$$1""8�
"�  "w��޷{"
f�?" ����" ��," ��,"
	 ������"
  "  "�  ",|赨�{���"��" *$$1""8�
"�  "w����{"
	 �Ș���"
  "  "�  "��Õ�{"
	 �Ș���"P*Lgradient_tape/sequential/densenet169/conv5_block25_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1408,1,1]"�  "����{"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block25_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1408]"�  "��͌�{"
	 �臝��"P*Lgradient_tape/sequential/densenet169/conv5_block25_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1408]"�  ",}���{���"��" *$$1""8�
"�  "wȟ��{"
	 ��ݪ��"
  "  "�  "w�ط�{"
	 ��Ө��"
  "  "�  "u����{"
	 �ʋ���"
  "  "�  "u����{"
	 ��Ҩ��"
  "  "�  ",z����{���"��" *$$1""8�
"�  "u����{"
	 ������"
  "  "�  ",z�à�{���"��" *$$1""8�
"�  "uؽ��{"
	 �臝��"
  "  "�  "-�����{�ł"��" *$$1""8�
"�  "w����{"
	 �����"
  "  "�  "�����{"
	 ��Ҩ��"E*Agradient_tape/sequential/densenet169/conv5_block24_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  ",~����{ȝ�"��" *$$1""8�
"�  "�����{"
	 ��Ԩ��"C*?gradient_tape/sequential/densenet169/conv5_block24_concat/Slice"
 �������"
*output" "*[200,1376,1,1]"�  ",~����{���"��" *$$1""8�
"�  "w����{"
	 �Ș���"
  "  "�  "�����{"
	 �Ș���"Y*Ugradient_tape/sequential/densenet169/conv5_block24_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�����{"
	 �ȡ���"Y*Ugradient_tape/sequential/densenet169/conv5_block24_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�����{"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block24_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  ",����{���"��" *$$1""8�
"�  ",R����{���"��" *$$1""8�
"�  "�ȝ��{"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block24_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ԭ�{"
	 �ȡ���"Y*Ugradient_tape/sequential/densenet169/conv5_block24_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��Լ�{"
	 �ȡ���"X*Tgradient_tape/sequential/densenet169/conv5_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �觩��"X*Tgradient_tape/sequential/densenet169/conv5_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  ",R����{��"��" *$$1""8�
"�  "����{"
	 �谩��"X*Tgradient_tape/sequential/densenet169/conv5_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S���|���"��" *$$1""8�
"�  "-��Ǖ�|�ݹ"��" *$$1""8�
"�  "-��綊|��"��" *$$1""8�
"�  "�����|"
	 �ȡ���"X*Tgradient_tape/sequential/densenet169/conv5_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����|"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��훎|"
	 �觩��"X*Tgradient_tape/sequential/densenet169/conv5_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w����|"
	 ��Ҩ��"
  "  "�  ",z��ǒ|���"��" *$$1""8�
"�  "w���|"
	 �Ș���"
  "  "�  ",|��|��"��" *$$1""8�
"�  "w����|"
	 ��̨��"
  "  "�  "�����|"
	 ��̨��"P*Lgradient_tape/sequential/densenet169/conv5_block24_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block24_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "��؜|"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block24_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  ",}���|���"��" *$$1""8�
"�  "wЖ�|"
	 �谩��"
  "  "�  "w��|"
	 ��Ũ��"
  "  "�  "u�弤|"
	 ������"
  "  "�  "u���|"
	 ������"
  "  "�  "�ذ��|"
	 �Ș���"Y*Ugradient_tape/sequential/densenet169/conv5_block24_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1376,128]"�  "����|"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block24_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1376,1,1]"�  "���Ϋ|"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block24_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2276]"�  "-�����|���"��" *$$1""8�
"�  "-�����|���"��" *$$1""8�
"�  "-����|���"��" *$$1""8�
"�  "-�����|���"��" *$$1""8�
"�  ",R�ݺ�|���"��" *$$1""8�
"�  "����|"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block24_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����|"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block24_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����|"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1376,1,1]"�  "�����|"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1376,1,1]"�  ",R����|�"��" *$$1""8�
"�  "�����|"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1376,1,1]"�  "�ȁ��|"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S����|���"��" *$$1""8�
"�  "-��އ�|���"��" *$$1""8�
"�  "-��Ů�|���"��" *$$1""8�
"�  "�����|"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����|"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����|"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w����|"
	 ��̨��"
  "  "�  ",z����|�ף"��" *$$1""8�
"�  "u���|"
	 ������"
  "  "�  ",z����|���"��" *$$1""8�
"�  "u����|"
	 �����"
  "  "�  ",zظ��|���"��" *$$1""8�
"�  "w���|"
	 �Ș���"
  "  "�  ",|����|�ˀ"��" *$$1""8�
"�  "w����|"
	 �Ȃ���"
  "  "�  "�І��|"
	 �Ȃ���"P*Lgradient_tape/sequential/densenet169/conv5_block24_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1376,1,1]"�  "�����|"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block24_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1376]"�  "�����|"
	 �臝��"P*Lgradient_tape/sequential/densenet169/conv5_block24_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1376]"�  ",}�ܼ�|軼"��" *$$1""8�
"�  "w����|"
	 �����"
  "  "�  "w����|"
	 ������"
  "  "�  "u����|"
	 ������"
  "  "�  "u����|"
	 �஦��"
  "  "�  ",z�މ�|��"��" *$$1""8�
"�  "u����|"
	 ������"
  "  "�  ",z��}ذ�"��" *$$1""8�
"�  "u����}"
	 �臝��"
  "  "�  "-���؅}�ب"��" *$$1""8�
"�  "w���}"
	 ��Ԩ��"
  "  "�  "�����}"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block23_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  ",~����}Я�"��" *$$1""8�
"�  "��ዏ}"
	 ������"C*?gradient_tape/sequential/densenet169/conv5_block23_concat/Slice"
 �������"
*output" "*[200,1344,1,1]"�  ",~ؗ��}��"��" *$$1""8�
"�  "w����}"
	 �Ȃ���"
  "  "�  "�ȗޓ}"
	 �Ȃ���"Y*Ugradient_tape/sequential/densenet169/conv5_block23_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�����}"
	 �ȋ���"Y*Ugradient_tape/sequential/densenet169/conv5_block23_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "���Ø}"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block23_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  ",�ȷ�}���"��" *$$1""8�
"�  ",R�ǽ�}���"��" *$$1""8�"�  "�ȳ�}"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block23_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����}"
	 �ȋ���"Y*Ugradient_tape/sequential/densenet169/conv5_block23_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�нê}"
	 �ȋ���"X*Tgradient_tape/sequential/densenet169/conv5_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �葨��"X*Tgradient_tape/sequential/densenet169/conv5_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  ",RЭ�}��"��" *$$1""8�"�  "�����}"
	 �蚨��"X*Tgradient_tape/sequential/densenet169/conv5_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S��}��"��" *$$1""8�"�  "-�𣣻}���"��" *$$1""8�"�  "-�����}���"��" *$$1""8�"�  "�����}"
	 �ȋ���"X*Tgradient_tape/sequential/densenet169/conv5_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��١�}"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����}"
	 �葨��"X*Tgradient_tape/sequential/densenet169/conv5_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w���}"
	 ������"
  "  "�  ",z����}���"��" *$$1""8�"�  "wб��}"
	 �Ȃ���"
  "  "�  ",|����}���"��" *$$1""8�"�  "w����}"
	 ������"
  "  "�  "����}"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block23_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block23_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "����}"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block23_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  ",}����}�̹"��" *$$1""8�"�  "w����}"
	 �蚨��"
  "  "�  "w����}"
	 ����"
  "  "�  "u����}"
	 ������"
  "  "�  "uأ��}"
	 ������"
  "  "�  "�����}"
	 �Ȃ���"Y*Ugradient_tape/sequential/densenet169/conv5_block23_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1344,128]"�  "�謰�}"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block23_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1344,1,1]"�  "�Ї��}"
	 �Ʈ���"Y*Ugradient_tape/sequential/densenet169/conv5_block23_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2276]"�  "-�����}���"��" *$$1""8�"�  "-��͓�}��"��" *$$1""8�"�  "-�����}��"��" *$$1""8�"�  "-�����}���"��" *$$1""8�"�  ",R����}���"��" *$$1""8�"�  "�����}"
	 �Ʈ���"Y*Ugradient_tape/sequential/densenet169/conv5_block23_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����}"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block23_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����}"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1344,1,1]"�  "��΂�}"
	 �ʔ���"X*Tgradient_tape/sequential/densenet169/conv5_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1344,1,1]"�  ",R����}���"��" *$$1""8�"�  "�����}"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1344,1,1]"�  "�����}"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",Sȅ��~�ɠ"��" *$$1""8�"�  "-�����~���"��" *$$1""8�"�  "-���ۈ~���"��" *$$1""8�"�  "�����~"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���ό~"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��~"
	 �ʔ���"X*Tgradient_tape/sequential/densenet169/conv5_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w���~"
	 ������"
  "  "�  ",z����~���"��" *$$1""8�"�  "u��ܔ~"
	 ������"
  "  "�  ",zФ��~���"��" *$$1""8�"�  "u����~"
	 �����"
  "  "�  ",z�Ŝ~���"��" *$$1""8�"�  "w����~"
	 �Ȃ���"
  "  "�  ",|��Ӡ~���"��" *$$1""8�"�  "w��Σ~"
	 �����"
  "  "�  "���ߤ~"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block23_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1344,1,1]"�  "����~"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block23_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1344]"�  "�����~"
	 �臝��"P*Lgradient_tape/sequential/densenet169/conv5_block23_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1344]"�  ",}��~��"��" *$$1""8�"�  "w����~"
	 ������"
  "  "�  "w��խ~"
	 �Я���"
  "  "�  "u����~"
	 ������"
  "  "�  "u�垮~"
	 ������"
  "  "�  ",z����~ȋ�"��" *$$1""8�"�  "u����~"
	 ������"
  "  "�  ",z��Ѷ~���"��" *$$1""8�"�  "uȇ��~"
	 �臝��"
  "  "�  "-���к~��"��" *$$1""8�"�  "w�ɟ�~"
	 ������"
  "  "�  "���Ҿ~"
	 �Я���"E*Agradient_tape/sequential/densenet169/conv5_block22_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  ",~���~耦"��" *$$1""8�"�  "�����~"
	 ������"C*?gradient_tape/sequential/densenet169/conv5_block22_concat/Slice"
 �������"
*output" "*[200,1312,1,1]"�  ",~���~���"��" *$$1""8�"�  "w����~"
	 �����"
  "  "�  "����~"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block22_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�����~"
	 ��ۨ��"Y*Ugradient_tape/sequential/densenet169/conv5_block22_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�ȴ��~"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block22_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  ",���~��"��" *$$1""8�"�  ",R����~���"��" *$$1""8�"�  "����~"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block22_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����~"
	 ��ۨ��"Y*Ugradient_tape/sequential/densenet169/conv5_block22_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�跾�~"
	 ��ۨ��"X*Tgradient_tape/sequential/densenet169/conv5_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  ",R����~���"��" *$$1""8�"�  "�����~"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S����~��"��" *$$1""8�"�  "-�����~���"��" *$$1""8�"�  "-�����~���"��" *$$1""8�"�  "��ބ�~"
	 ��ۨ��"X*Tgradient_tape/sequential/densenet169/conv5_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����~"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�г��~"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w����~"
	 �Я���"
  "  "�  ",z�ۍ�~���"��" *$$1""8�"�  "w�ݺ�~"
	 ��Ҩ��"
  "  "�  ",|褋�Ȕ�"��" *$$1""8�"�  "w؉�"
	 �触��"
  "  "�  "�����"
	 �触��"P*Lgradient_tape/sequential/densenet169/conv5_block22_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block22_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "�����"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block22_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  ",}ة����"��" *$$1""8�"�  "w����"
	 �����"
  "  "�  "w���"
	 �ȡ���"
  "  "�  "u��"
	 ������"
  "  "�  "u����"
	 ������"
  "  "�  "�س��"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block22_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1312,128]"�  "��ؚ�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block22_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1312,1,1]"�  "�讜�"
	 �Ʈ���"Y*Ugradient_tape/sequential/densenet169/conv5_block22_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2276]"�  "-�؂՛���"��" *$$1""8�"�  "-���ǟ���"��" *$$1""8�"�  "-��à����"��" *$$1""8�"�  "-��������"��" *$$1""8�"�  ",R��ާ���"��" *$$1""8�"�  "�����"
	 �Ʈ���"Y*Ugradient_tape/sequential/densenet169/conv5_block22_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�И�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block22_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1312,1,1]"�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1312,1,1]"�  ",R�Ϝ���"��" *$$1""8�"�  "�����"
	 ��ˬ��"X*Tgradient_tape/sequential/densenet169/conv5_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1312,1,1]"�  "��遶"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  ",S�؅���"��" *$$1""8�"�  "-��뼽���"��" *$$1""8�"�  "-���ٿ���"��" *$$1""8�"�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�П��"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�ȏ��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "w����"
	 �触��"
  "  "�  ",z����п�"��" *$$1""8�"�  "u�ԟ�"
	 ������"
  "  "�  ",z�������"��" *$$1""8�"�  "u����"
	 �����"
  "  "�  ",z�������"��" *$$1""8�"�  "w����"
	 ��Ҩ��"
  "  "�  ",|�������"��" *$$1""8�"�  "w؎��"
	 �����"
  "  "�  "�෢�"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block22_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1312,1,1]"�  "�����"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block22_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1312]"�  "����"
	 �臝��"P*Lgradient_tape/sequential/densenet169/conv5_block22_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1312]"�  ",}������"��" *$$1""8�"�  "w����"
	 ��ˬ��"
  "  "�  "w����"
	 ������"
  "  "�  "u����"
	 ������"
  "  "�  "u���"
	 �ږ���"
  "  "�  ",z�������"��" *$$1""8�"�  "u����"
	 ������"
  "  "�  ",z�������"��" *$$1""8�"�  "u�ם�"
	 �臝��"
  "  "�  "-��������"��" *$$1""8�"�  "x��с�"
	 ������"
  "  "�  "���փ�"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block21_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~�汅��¢"��" *$$1""8�"�  "���늀"
	 ������"C*?gradient_tape/sequential/densenet169/conv5_block21_concat/Slice"
 �������"
*output" "*[200,1280,1,1]"�  "-~��Ŏ����"��" *$$1""8�"�  "x蟤��"
	 �����"
  "  "�  "������"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block21_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "���Ɩ�"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block21_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block21_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-Ў�����"��" *$$1""8�"�  "-R��鬀���"��" *$$1""8�"�  "��͡ր"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block21_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����׀"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block21_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��Ǌڀ"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R���݀�ď"��" *$$1""8�"�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�������"��" *$$1""8�"�  ".��݀��Ǹ"��" *$$1""8�"�  ".�Б�����"��" *$$1""8�"�  "��°�"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x����"
	 ������"
  "  "�  "-z����ؔ�"��" *$$1""8�"�  "x�����"
	 �����"
  "  "�  "-|�����শ"��" *$$1""8�"�  "x��ǂ�"
	 ������"
  "  "�  "���Ճ�"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block21_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block21_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "���ф�"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block21_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}��������"��" *$$1""8�"�  "x��׌�"
	 ������"
  "  "�  "x�����"
	 ������"
  "  "�  "v�ϯ��"
	 ������"
  "  "�  "v�׍�"
	 ������"
  "  "�  "������"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block21_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1280,128]"�  "����"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block21_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1280,1,1]"�  "������"
	 �Ʈ���"Y*Ugradient_tape/sequential/densenet169/conv5_block21_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2268]"�  ".���Κ����"��" *$$1""8�"�  ".��ِ�����"��" *$$1""8�"�  ".��Ҧ�����"��" *$$1""8�"�  ".���ۣ���"��" *$$1""8�"�  "-R��������"��" *$$1""8�"�  "������"
	 �Ʈ���"Y*Ugradient_tape/sequential/densenet169/conv5_block21_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��Ě��"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block21_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1280,1,1]"�  "�𔫮�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1280,1,1]"�  "-R�Щ���ϫ"��" *$$1""8�"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1280,1,1]"�  "��򭶁"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S𗆺��"��" *$$1""8�"�  ".��֬��Ȝ�"��" *$$1""8�"�  ".�؎п����"��" *$$1""8�"�  "�ؕ�"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���Á"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�χā"
	 ������"
  "  "�  "-z���ǁ��"��" *$$1""8�"�  "v���ˁ"
	 ������"
  "  "�  "-zȶ�́���"��" *$$1""8�"�  "v���Ё"
	 �����"
  "  "�  "-z���ҁࣷ"��" *$$1""8�"�  "xد�ց"
	 �����"
  "  "�  "-|���ׁ��"��" *$$1""8�"�  "x���ځ"
	 ��֤��"
  "  "�  "����ہ"
	 ��֤��"P*Lgradient_tape/sequential/densenet169/conv5_block21_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1280,1,1]"�  "����܁"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block21_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1280]"�  "����܁"
	 �臝��"P*Lgradient_tape/sequential/densenet169/conv5_block21_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1280]"�  "-}�����М�"��" *$$1""8�"�  "x����"
	 ������"
  "  "�  "x���"
	 ������"
  "  "�  "v����"
	 ������"
  "  "�  "v����"
	 �苢��"
  "  "�  "-z����Н�"��" *$$1""8�"�  "v����"
	 ������"
  "  "�  "-z�������"��" *$$1""8�"�  "v����"
	 �臝��"
  "  "�  ".�輘�蟀'"��" *$$1""8�"�  "x��ʙ�"
	 ������"
  "  "�  "���˛�"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block20_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~������ׁ"��" *$$1""8�"�  "�Ъᡂ"
	 ������"C*?gradient_tape/sequential/densenet169/conv5_block20_concat/Slice"
 �������"
*output" "*[200,1248,1,1]"�  "-~��ࢂ���"��" *$$1""8�"�  "x��ĥ�"
	 ��֤��"
  "  "�  "���ꦂ"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block20_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "���ڨ�"
	 ��ۨ��"Y*Ugradient_tape/sequential/densenet169/conv5_block20_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block20_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-�������"��" *$$1""8�"�  "-R��������"��" *$$1""8�"�  "���ݹ�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block20_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�ؠк�"
	 ��ۨ��"Y*Ugradient_tape/sequential/densenet169/conv5_block20_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���ټ�"
	 ��ۨ��"X*Tgradient_tape/sequential/densenet169/conv5_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R�ן�����"��" *$$1""8�"�  "����Â"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
�+��?" ����" �" � "
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-Sತ˂���"��" *$$1""8�"�  ".�ض�΂���"��" *$$1""8�"�  ".�Ђ�тȥ�"��" *$$1""8�"�  "����ӂ"
	 ��ۨ��"X*Tgradient_tape/sequential/densenet169/conv5_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ёԂ"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����Ԃ"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�ߦՂ"
	 ������"
  "  "�  "-z���ق���"��" *$$1""8�"�  "x��܂"
	 ��Ҩ��"
  "  "�  "-|贡ނ�מ"��" *$$1""8�"�  "x���"
	 ������"
  "  "�  "�؛��"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block20_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block20_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "�ح��"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block20_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}�������"��" *$$1""8�"�  "x����"
	 �����"
  "  "�  "x����"
	 ������"
  "  "�  "v����"
	 ������"
  "  "�  "v����"
	 ������"
  "  "�  "�����"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block20_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1248,128]"�  "�����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block20_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1248,1,1]"�  "��խ�"
	 �Ʈ���"Y*Ugradient_tape/sequential/densenet169/conv5_block20_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2268]"�  ".�Ъ���ط�"��" *$$1""8�"�  ".��Ϗ�����"��" *$$1""8�"�  ".�������ܗ"��" *$$1""8�"�  ".���䁃���"��" *$$1""8�"�  "-R��������"��" *$$1""8�"�  "��β��"
	 �Ʈ���"Y*Ugradient_tape/sequential/densenet169/conv5_block20_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block20_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1248,1,1]"�  "���ދ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1248,1,1]"�  "-R��Ό��٘"��" *$$1""8�"�  "���ŏ�"
	 ��ɬ��"X*Tgradient_tape/sequential/densenet169/conv5_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1248,1,1]"�  "������"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S��ᖃ�Ť"��" *$$1""8�"�  ".�؏������"��" *$$1""8�"�  ".��è�����"��" *$$1""8�"�  "����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���ן�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��Ǡ�"
	 ������"
  "  "�  "-z�����Н�"��" *$$1""8�"�  "v�Ǫ��"
	 ������"
  "  "�  "-z�����"��" *$$1""8�"�  "v��Ǭ�"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "x�����"
	 ��Ҩ��"
  "  "�  "-|�ř��Є�"��" *$$1""8�"�  "x��ƶ�"
	 ��̣��"
  "  "�  "�ȥ޷�"
	 ��̣��"P*Lgradient_tape/sequential/densenet169/conv5_block20_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1248,1,1]"�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block20_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1248]"�  "���۸�"
	 �臝��"P*Lgradient_tape/sequential/densenet169/conv5_block20_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1248]"�  "-}��������"��" *$$1""8�"�  "x��Ϳ�"
	 ��ɬ��"
  "  "�  "x�����"
	 ������"
  "  "�  "v�ݦ��"
	 �ڍ���"
  "  "�  "v�����"
	 ������"
  "  "�  "-z���Ã�͊"��" *$$1""8�"�  "v���ƃ"
	 ������"
  "  "�  "-z���ȃ���"��" *$$1""8�"�  "v���˃"
	 �臝��"
  "  "�  ".����̃���"��" *$$1""8�"�  "x��σ"
	 ������"
  "  "�  "�в�Ѓ"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block19_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~���҃���"��" *$$1""8�"�  "����Ճ"
	 �ȑ���"C*?gradient_tape/sequential/densenet169/conv5_block19_concat/Slice"
 �������"
*output" "*[200,1216,1,1]"�  "-~���փ���"��" *$$1""8�"�  "x���ك"
	 ��̣��"
  "  "�  "���ڃ"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block19_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "��Ԟ܃"
	 ��ۨ��"Y*Ugradient_tape/sequential/densenet169/conv5_block19_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "����߃"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block19_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-������"��" *$$1""8�"�  "-R������"��" *$$1""8�"�  "�����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block19_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ��ۨ��"Y*Ugradient_tape/sequential/densenet169/conv5_block19_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�ا���"
	 ��ۨ��"X*Tgradient_tape/sequential/densenet169/conv5_block19_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block19_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R������"��" *$$1""8�"�  "������"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block19_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block19_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�������"��" *$$1""8�"�  ".�𢿁����"��" *$$1""8�"�  ".���惄���"��" *$$1""8�"�  "�ؕ���"
	 ��ۨ��"X*Tgradient_tape/sequential/densenet169/conv5_block19_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�𦂇�"
	 �Ʈ���"X*Tgradient_tape/sequential/densenet169/conv5_block19_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��谇�"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block19_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 ������"
  "  "�  "-z��ߋ����"��" *$$1""8�"�  "x�����"
	 ��Ҩ��"
  "  "�  "-|�����螜"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block19_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block19_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "��ゖ�"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block19_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}�����ȇ�"��" *$$1""8�"�  "x�憝�"
	 �����"
  "  "�  "x�񱝄"
	 ������"
  "  "�  "v��ҝ�"
	 ������"
  "  "�  "v�����"
	 ������"
  "  "�  "��ޢ��"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block19_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1216,128]"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block19_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1216,1,1]"�  "���٤�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block19_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2268]"�  ".�؟������"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  "-R������٠"��" *$$1""8�"�  "��٤��"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block19_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block19_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block19_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1216,1,1]"�  "���ż�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block19_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1216,1,1]"�  "-R�Ƹ��ؾ�"��" *$$1""8�"�  "��Ƽ��"
	 ��Ȭ��"X*Tgradient_tape/sequential/densenet169/conv5_block19_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1216,1,1]"�  "�Н�Ą"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block19_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S���Ǆ�Ǚ"��" *$$1""8�"�  ".����ʄ���"��" *$$1""8�"�  ".�Д�̄���"��" *$$1""8�"�  "��վτ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block19_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���Є"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block19_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���Є"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block19_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x���ф"
	 ������"
  "  "�  "-zо�Ԅ���"��" *$$1""8�"�  "v���ׄ"
	 ������"
  "  "�  "-z�ڄ���"��" *$$1""8�"�  "v��݄"
	 �����"
  "  "�  "-z���߄ȕ�"��" *$$1""8�"�  "x����"
	 ��Ҩ��"
  "  "�  "-|������"��" *$$1""8�"�  "x�̸�"
	 ��Ȣ��"
  "  "�  "�����"
	 �؃���"P*Lgradient_tape/sequential/densenet169/conv5_block19_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1216,1,1]"�  "��ڎ�"
	 �ڄ���"P*Lgradient_tape/sequential/densenet169/conv5_block19_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1216]"�  "��ĺ�"
	 �ڍ���"P*Lgradient_tape/sequential/densenet169/conv5_block19_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1216]"�  "-}�������"��" *$$1""8�"�  "x���"
	 ��Ȭ��"
  "  "�  "x����"
	 ������"
  "  "�  "v����"
	 �挠��"
  "  "�  "vྜ��"
	 ������"
  "  "�  "-z؝����"��" *$$1""8�"�  "v�����"
	 �ڄ���"
  "  "�  "-z��������"��" *$$1""8�"�  "v����"
	 �ڍ���"
  "  "�  ".���������"��" *$$1""8�"�  "x�����"
	 �ȑ���"
  "  "�  "�Ы���"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block18_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~��������"��" *$$1""8�"�  "���ᅅ"
	 ����"C*?gradient_tape/sequential/densenet169/conv5_block18_concat/Slice"
 �������"
*output" "*[200,1184,1,1]"�  "-~��������"��" *$$1""8�"�  "x�󥉅"
	 �؃���"
  "  "�  "������"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block18_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "����"
	 �،���"Y*Ugradient_tape/sequential/densenet169/conv5_block18_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�п���"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block18_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-�ؾ�����"��" *$$1""8�"�  "-R��ܛ����"��" *$$1""8�"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block18_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����"
	 �،���"Y*Ugradient_tape/sequential/densenet169/conv5_block18_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 �،���"X*Tgradient_tape/sequential/densenet169/conv5_block18_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block18_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R��������"��" *$$1""8�"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block18_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block18_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S��Э��"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".�聛�����"��" *$$1""8�"�  "������"
	 �،���"X*Tgradient_tape/sequential/densenet169/conv5_block18_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ŷ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block18_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block18_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��׸�"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "x�����"
�2��?" ����" ��	" ��	"
	 �؃���"
  "  "�  "-|�����"��" *$$1""8�"�  "x��ą"
	 ������"
  "  "�  "��֏Ņ"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block18_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block18_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "����ƅ"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block18_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}��Ʌ�Ĵ"��" *$$1""8�"�  "x���ͅ"
	 ������"
  "  "�  "x�ȵͅ"
	 ������"
  "  "�  "v���ͅ"
	 ������"
  "  "�  "v���΅"
	 ������"
  "  "�  "����υ"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block18_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1184,128]"�  "�Ю�х"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block18_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1184,1,1]"�  "����ԅ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block18_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2268]"�  ".���څ�ɺ"��" *$$1""8�"�  ".����݅���"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  "-R�������"��" *$$1""8�"�  "�����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block18_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block18_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block18_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1184,1,1]"�  "�����"
	 �〈��"X*Tgradient_tape/sequential/densenet169/conv5_block18_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1184,1,1]"�  "-R�Б��"��" *$$1""8�"�  "�б���"
	 �Я���"X*Tgradient_tape/sequential/densenet169/conv5_block18_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1184,1,1]"�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block18_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�̕�����"��" *$$1""8�"�  ".��Ǎ�����"��" *$$1""8�"�  ".���ʀ����"��" *$$1""8�"�  "��Ψ��"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block18_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block18_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��յ��"
	 �〈��"X*Tgradient_tape/sequential/densenet169/conv5_block18_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��"
	 ������"
  "  "�  "-z諅���׸"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-z��ꎆ���"��" *$$1""8�"�  "vȜ���"
	 �����"
  "  "�  "-z��������"��" *$$1""8�"�  "x�ì��"
	 �؃���"
  "  "�  "-|��������"��" *$$1""8�"�  "xЉݚ�"
	 ��ġ��"
  "  "�  "�Ȏ���"
	 �؃���"P*Lgradient_tape/sequential/densenet169/conv5_block18_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1184,1,1]"�  "�Ȓǜ�"
	 �ڄ���"P*Lgradient_tape/sequential/densenet169/conv5_block18_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1184]"�  "�й���"
	 �ڍ���"P*Lgradient_tape/sequential/densenet169/conv5_block18_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1184]"�  "-}أ���ȿ�"��" *$$1""8�"�  "x�����"
	 �Я���"
  "  "�  "x��Ѧ�"
	 ������"
  "  "�  "v�����"
	 �ކ���"
  "  "�  "v�����"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "v蓞��"
	 �ڄ���"
  "  "�  "-z��ѯ��ۼ"��" *$$1""8�"�  "v��粆"
	 �ڍ���"
  "  "�  ".�ظ����܅"��" *$$1""8�"�  "x�����"
	 ����"
  "  "�  "���ĸ�"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block17_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~��ٹ����"��" *$$1""8�"�  "���쾆"
	 ����"C*?gradient_tape/sequential/densenet169/conv5_block17_concat/Slice"
 �������"
*output" "*[200,1152,1,1]"�  "-~������͊"��" *$$1""8�"�  "x���"
	 �؃���"
  "  "�  "����Æ"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block17_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "����ņ"
	 �،���"Y*Ugradient_tape/sequential/densenet169/conv5_block17_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "����Ɇ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block17_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-貁φ���"��" *$$1""8�"�  "-Rإ�Ԇ���"��" *$$1""8�"�  "��ξֆ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block17_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ڢ׆"
	 �،���"Y*Ugradient_tape/sequential/densenet169/conv5_block17_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����ن"
	 �،���"X*Tgradient_tape/sequential/densenet169/conv5_block17_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block17_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R���݆���"��" *$$1""8�"�  "��ڶ��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block17_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block17_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S蔽����"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  "�����"
	 �،���"X*Tgradient_tape/sequential/densenet169/conv5_block17_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block17_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block17_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x���"
	 ������"
  "  "�  "-z�������"��" *$$1""8�"�  "x�ϯ��"
	 �؃���"
  "  "�  "-|�������"��" *$$1""8�"�  "x蜜��"
	 �脡��"
  "  "�  "������"
	 �؃���"P*Lgradient_tape/sequential/densenet169/conv5_block17_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block17_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "���냇"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block17_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}�ɒ��خ�"��" *$$1""8�"�  "xȎ���"
	 ������"
  "  "�  "x��Ջ�"
	 ������"
  "  "�  "v���"
	 ������"
  "  "�  "v�����"
	 ������"
  "  "�  "���ɍ�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block17_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1152,128]"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block17_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1152,1,1]"�  "���ϔ�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block17_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2260]"�  ".���������"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".���Π����"��" *$$1""8�"�  ".�؈����"��" *$$1""8�"�  "-R��������"��" *$$1""8�"�  "���Ĩ�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block17_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block17_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block17_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1152,1,1]"�  "���鬇"
	 �趡��"X*Tgradient_tape/sequential/densenet169/conv5_block17_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1152,1,1]"�  "-R��ݭ����"��" *$$1""8�"�  "������"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block17_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1152,1,1]"�  "���ݴ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block17_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�񨸇���"��" *$$1""8�"�  ".�ط»����"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block17_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��тÇ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block17_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����Ç"
	 �趡��"X*Tgradient_tape/sequential/densenet169/conv5_block17_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x���ć"
	 �؃���"
  "  "�  "-zؖ�ȇ���"��" *$$1""8�"�  "vЯ�ˇ"
	 ������"
  "  "�  "-z���·���"��" *$$1""8�"�  "v���ч"
	 �����"
  "  "�  "-z���ԇ���"��" *$$1""8�"�  "x���և"
	 ������"
  "  "�  "-|���؇���"��" *$$1""8�"�  "x�Ǟۇ"
	 ��Ơ��"
  "  "�  "����݇"
	 �؃���"P*Lgradient_tape/sequential/densenet169/conv5_block17_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1152,1,1]"�  "����އ"
	 �ڄ���"P*Lgradient_tape/sequential/densenet169/conv5_block17_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1152]"�  "����އ"
	 �ڍ���"P*Lgradient_tape/sequential/densenet169/conv5_block17_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1152]"�  "-}�������"��" *$$1""8�"�  "x����"
	 ��Ҩ��"
  "  "�  "x�٫�"
	 ������"
  "  "�  "v����"
	 ������"
  "  "�  "v����"
	 ������"
  "  "�  "-z���Ь�"��" *$$1""8�"�  "v����"
	 �ڄ���"
  "  "�  "-z�������"��" *$$1""8�"�  "v����"
	 �ڍ���"
  "  "�  ".�����،�"��" *$$1""8�"�  "x�����"
	 ����"
  "  "�  "������"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block16_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~�������"��" *$$1""8�"�  "������"
	 ����"C*?gradient_tape/sequential/densenet169/conv5_block16_concat/Slice"
 �������"
*output" "*[200,1120,1,1]"�  "-~��������"��" *$$1""8�"�  "x��߂�"
	 �؃���"
  "  "�  "����"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block16_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "���Ӆ�"
	 �،���"Y*Ugradient_tape/sequential/densenet169/conv5_block16_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "���䈈"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block16_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-�㎈���"��" *$$1""8�"�  "-R��ԓ�г�"��" *$$1""8�"�  "����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block16_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 �،���"Y*Ugradient_tape/sequential/densenet169/conv5_block16_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 �،���"X*Tgradient_tape/sequential/densenet169/conv5_block16_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block16_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R����Ͷ"��" *$$1""8�"�  "��؏��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block16_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block16_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S��������"��" *$$1""8�"�  ".���⫈��"��" *$$1""8�"�  ".�В����ƹ"��" *$$1""8�"�  "���Ȱ�"
	 �،���"X*Tgradient_tape/sequential/densenet169/conv5_block16_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ד��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block16_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���±�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block16_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x����"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "x��㺈"
	 �؃���"
  "  "�  "-|��Ǽ����"��" *$$1""8�"�  "x��ӿ�"
	 ������"
  "  "�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block16_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block16_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "������"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block16_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}���Ĉ���"��" *$$1""8�"�  "x�Ɉ"
	 ������"
  "  "�  "x���Ɉ"
	 ������"
  "  "�  "v���ʈ"
	 ������"
  "  "�  "v�ɘˈ"
	 ������"
  "  "�  "����̈"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block16_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1120,128]"�  "����Έ"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block16_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1120,1,1]"�  "����҈"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block16_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2260]"�  ".����؈���"��" *$$1""8�"�  ".����܈���"��" *$$1""8�"�  ".����ވ���"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  "-R�������"��" *$$1""8�"�  "�����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block16_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block16_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�ر��"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block16_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1120,1,1]"�  "��ͮ�"
	 �҉���"X*Tgradient_tape/sequential/densenet169/conv5_block16_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1120,1,1]"�  "-R�������"��" *$$1""8�"�  "������"
	 �Ҭ���"X*Tgradient_tape/sequential/densenet169/conv5_block16_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1120,1,1]"�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block16_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S��������"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  "��씀�"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block16_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���ۀ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block16_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��爁�"
	 �҉���"X*Tgradient_tape/sequential/densenet169/conv5_block16_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��灉"
	 ������"
  "  "�  "-z��Ņ����"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "v�����"
	 �����"
  "  "�  "-z��푉���"��" *$$1""8�"�  "x�Ҵ��"
	 �؃���"
  "  "�  "-|���ħ"��" *$$1""8�"�  "x�����"
	 �̴���"
  "  "�  "��߈��"
	 �؃���"P*Lgradient_tape/sequential/densenet169/conv5_block16_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1120,1,1]"�  "���ٜ�"
	 �ڄ���"P*Lgradient_tape/sequential/densenet169/conv5_block16_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1120]"�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block16_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1120]"�  "-}��������"��" *$$1""8�"�  "x�����"
	 �Ҭ���"
  "  "�  "x��ͤ�"
	 ������"
  "  "�  "v��契"
	 ������"
  "  "�  "v�魦�"
	 �̅���"
  "  "�  "-z��é����"��" *$$1""8�"�  "v��ͬ�"
	 �ڄ���"
  "  "�  "-z𓊯����"��" *$$1""8�"�  "v�̱�"
	 ������"
  "  "�  ".�о�����"��" *$$1""8�"�  "x��浉"
	 ����"
  "  "�  "��璷�"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block15_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~��������"��" *$$1""8�"�  "������"
	 ������"C*?gradient_tape/sequential/densenet169/conv5_block15_concat/Slice"
 �������"
*output" "*[200,1088,1,1]"�  "-~������Ӽ"��" *$$1""8�"�  "x����"
	 �؃���"
  "  "�  "������"
	 �ȁ���"Y*Ugradient_tape/sequential/densenet169/conv5_block15_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "����ĉ"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block15_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "��նǉ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block15_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-���͉���"��" *$$1""8�"�  "-R���ԉ���"��" *$$1""8�"�  "����։"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block15_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����׉"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block15_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ى"
	 �؃���"X*Tgradient_tape/sequential/densenet169/conv5_block15_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block15_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-RȞ�ۉ��"��" *$$1""8�"�  "����߉"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block15_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block15_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�������"��" *$$1""8�"�  ".����д�"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  "�肊�"
	 �؃���"X*Tgradient_tape/sequential/densenet169/conv5_block15_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block15_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�莃�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block15_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x����"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "xأ���"
	 �ȁ���"
  "  "�  "-|�ݩ�����"��" *$$1""8�"�  "x����"
	 ������"
  "  "�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block15_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block15_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "���₊"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block15_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}��������"��" *$$1""8�"�  "x𛚊�"
	 ������"
  "  "�  "x��֊�"
	 �����"
  "  "�  "v؀���"
	 ������"
  "  "�  "v�����"
	 ������"
  "  "�  "���Ό�"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block15_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1088,128]"�  "������"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block15_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1088,1,1]"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block15_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2260]"�  ".���䗊��"��" *$$1""8�"�  ".���益���"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".���ڠ�Њ�"��" *$$1""8�"�  "-R��������"��" *$$1""8�"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block15_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���稊"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block15_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block15_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1088,1,1]"�  "��ط��"
	 �Ħ���"X*Tgradient_tape/sequential/densenet169/conv5_block15_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1088,1,1]"�  "-RЅ����Ǟ"��" *$$1""8�"�  "��禰�"
	 ��ȟ��"X*Tgradient_tape/sequential/densenet169/conv5_block15_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1088,1,1]"�  "��ۖ��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block15_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S��巊�Ǧ"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".��Ϧ�����"��" *$$1""8�"�  "���뿊"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block15_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block15_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�и���"
	 �Ħ���"X*Tgradient_tape/sequential/densenet169/conv5_block15_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��Ê"
	 ������"
  "  "�  "-z���Ɗ�λ"��" *$$1""8�"�  "v���ʊ"
	 ������"
  "  "�  "-zȴ�̊�ˢ"��" *$$1""8�"�  "v��ϊ"
	 �����"
  "  "�  "-zȁ�Ҋ���"��" *$$1""8�"�  "xЯ�Պ"
	 �؃���"
  "  "�  "-|���֊ȗ�"��" *$$1""8�"�  "x��ي"
	 ������"
  "  "�  "����ڊ"
	 �؃���"P*Lgradient_tape/sequential/densenet169/conv5_block15_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1088,1,1]"�  "�蚾ۊ"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block15_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1088]"�  "����݊"
	 �ڄ���"P*Lgradient_tape/sequential/densenet169/conv5_block15_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1088]"�  "-}��������"��" *$$1""8�"�  "x����"
	 ��ȟ��"
  "  "�  "x�ߑ�"
	 ������"
  "  "�  "vȳ��"
	 ������"
  "  "�  "v����"
	 ������"
  "  "�  "-z�����ӕ"��" *$$1""8�"�  "v����"
	 ������"
  "  "�  "-z�ۮ���"��" *$$1""8�"�  "v���"
	 �ڄ���"
  "  "�  ".�л����"��" *$$1""8�"�  "x����"
	 ������"
  "  "�  "��˽��"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block14_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~�������"��" *$$1""8�"�  "������"
	 ������"C*?gradient_tape/sequential/densenet169/conv5_block14_concat/Slice"
 �������"
*output" "*[200,1056,1,1]"�  "-~�������"��" *$$1""8�"�  "x��Ɓ�"
	 �؃���"
  "  "�  "���݂�"
	 �ȁ���"Y*Ugradient_tape/sequential/densenet169/conv5_block14_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "���Ƅ�"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block14_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "���㇋"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block14_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-Ȱ������"��" *$$1""8�"�  "-RІɒ����"��" *$$1""8�"�  "��ꡖ�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block14_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ŋ��"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block14_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��⼚�"
	 �؃���"X*Tgradient_tape/sequential/densenet169/conv5_block14_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block14_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R�����辻"��" *$$1""8�"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block14_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block14_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�ϛ�����"��" *$$1""8�"�  ".��ſ�����"��" *$$1""8�"�  ".�������"��" *$$1""8�
	 �؃���"X*Tgradient_tape/sequential/densenet169/conv5_block14_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block14_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ऱ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block14_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 ������"
  "  "�  "-z�絋�Ĵ"��" *$$1""8�
	 �ȁ���"
  "  "�  "-|��ﺋв�"��" *$$1""8�
	 ������"
  "  "�  "�ؠ���"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block14_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block14_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block14_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}���ċ���"��" *$$1""8�
	 ������"
  "  "�  "x�ʧɋ"
	 ������"
  "  "�  "vЎ�ɋ"
	 ������"
  "  "�  "v���ɋ"
	 ������"
  "  "�  "����ˋ"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block14_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1056,128]"�  "��ȉ΋"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block14_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1056,1,1]"�  "����ҋ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block14_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2260]"�  ".����׋Г�"��" *$$1""8�
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block14_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��Ї�"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block14_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ے�"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block14_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1056,1,1]"�  "����"
	 ����"X*Tgradient_tape/sequential/densenet169/conv5_block14_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1056,1,1]"�  "-R�������"��" *$$1""8�
	 �§���"X*Tgradient_tape/sequential/densenet169/conv5_block14_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1056,1,1]"�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block14_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S��������"��" *$$1""8�
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block14_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�آЀ�"
�8��?" ����" �" �"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block14_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ����"X*Tgradient_tape/sequential/densenet169/conv5_block14_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��ځ�"
	 ������"
  "  "�  "-z������ߞ"��" *$$1""8�
	 ������"
  "  "�  "-z��������"��" *$$1""8�
	 ������"
  "  "�  "-z��������"��" *$$1""8�
	 �؃���"
  "  "�  "-|�ȱ�����"��" *$$1""8�
	 ��ŝ��"
  "  "�  "���Ě�"
	 �؃���"P*Lgradient_tape/sequential/densenet169/conv5_block14_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1056,1,1]"�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block14_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1056]"�  "������"
	 �ڄ���"P*Lgradient_tape/sequential/densenet169/conv5_block14_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1056]"�  "-}��ޞ�ؽ�"��" *$$1""8�
	 �§���"
  "  "�  "x؞���"
	 ������"
  "  "�  "v�Լ��"
	 ��ǚ��"
  "  "�  "v��磌"
	 ������"
  "  "�  "-z�Ȓ���"��" *$$1""8�
	 ������"
  "  "�  "-zظ�����"��" *$$1""8�
	 �ڄ���"
  "  "�  ".��������"��" *$$1""8�
	 ������"
  "  "�  "��⻌"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block13_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~��������"��" *$$1""8�
	 ������"C*?gradient_tape/sequential/densenet169/conv5_block13_concat/Slice"
 �������"
*output" "*[200,1024,1,1]"�  "-~��������"��" *$$1""8�
	 �؃���"
  "  "�  "����Ō"
	 �ȁ���"Y*Ugradient_tape/sequential/densenet169/conv5_block13_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "��ؚǌ"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block13_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "��϶ʌ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block13_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-���ь���"��" *$$1""8�
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block13_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����ڌ"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block13_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����܌"
	 �؃���"X*Tgradient_tape/sequential/densenet169/conv5_block13_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block13_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R覜ߌ�Ѡ"��" *$$1""8�
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block13_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block13_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S������"��" *$$1""8�
	 �؃���"X*Tgradient_tape/sequential/densenet169/conv5_block13_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block13_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block13_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 ������"
  "  "�  "-zȡ�����"��" *$$1""8�
*�~��?" ����" ��	" ��"
	 �ȁ���"
  "  "�  "-|��������"��" *$$1""8�
	 �܋���"
  "  "�  "��П��"
*�~��?" ����" ��" ��"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block13_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block13_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "��ޕ��"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block13_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}Ў������"��" *$$1""8�
	 ������"
  "  "�  "xش���"
	 ������"
  "  "�  "v�º��"
	 ���"
  "  "�  "v��鍍"
	 ����"
  "  "�  "�ഡ��"
	 �؃���"Y*Ugradient_tape/sequential/densenet169/conv5_block13_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1024,128]"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block13_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1024,1,1]"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block13_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2252]"�  ".���Ԛ����"��" *$$1""8�
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block13_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block13_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���ݭ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block13_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1024,1,1]"�  "�軟��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block13_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1024,1,1]"�  "-R�蚰����"��" *$$1""8�
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block13_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1024,1,1]"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block13_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�����"��" *$$1""8�
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block13_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����č"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block13_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ͽč"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block13_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�ڡō"
	 ������"
  "  "�  "-z���ȍ��"��" *$$1""8�
	 ������"
  "  "�  "-z���ύ���"��" *$$1""8�
	 ������"
  "  "�  "-zд�Ս���"��" *$$1""8�
	 �؃���"
  "  "�  "-|���ٍ譗"��" *$$1""8�
	 ��Ӝ��"
  "  "�  "��Ͻݍ"
	 �؃���"P*Lgradient_tape/sequential/densenet169/conv5_block13_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1024,1,1]"�  "����ލ"
	 �Ʈ���"P*Lgradient_tape/sequential/densenet169/conv5_block13_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1024]"�  "�ਲލ"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block13_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1024]"�  "-}�������"��" *$$1""8�
	 ��Ҩ��"
  "  "�  "x����"
	 ������"
  "  "�  "v�̐�"
	 ��ǚ��"
  "  "�  "v�Է�"
	 ��ǚ��"
  "  "�  "-z������"��" *$$1""8�
	 �Ʈ���"
  "  "�  "-z������"��" *$$1""8�
	 ������"
  "  "�  ".�Ѕ������"��" *$$1""8�
	 ������"
  "  "�  "������"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block12_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~������"��" *$$1""8�
	 ������"C*?gradient_tape/sequential/densenet169/conv5_block12_concat/Slice"
 �������"
*output" "*
	 �؃���"
  "  "�  "���ʄ�"
	 �ȁ���"Y*Ugradient_tape/sequential/densenet169/conv5_block12_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "������"
	 ��Ҭ��"Y*Ugradient_tape/sequential/densenet169/conv5_block12_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "��ŉ�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block12_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-ؼ������"��" *$$1""8�
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block12_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���ޗ�"
	 ��Ҭ��"Y*Ugradient_tape/sequential/densenet169/conv5_block12_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���ܙ�"
	 ��Ҭ��"X*Tgradient_tape/sequential/densenet169/conv5_block12_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ��٬��"X*Tgradient_tape/sequential/densenet169/conv5_block12_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R��휎���"��" *$$1""8�
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block12_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block12_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S��������"��" *$$1""8�
	 ��Ҭ��"X*Tgradient_tape/sequential/densenet169/conv5_block12_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����Վ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block12_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��Ͳ֎"
	 ��٬��"X*Tgradient_tape/sequential/densenet169/conv5_block12_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�ٙ׎"
	 ������"
  "  "�  "-z���ێ���"��" *$$1""8�
	 �ȁ���"
  "  "�  "-|г������"��" *$$1""8�
	 �ԙ���"
  "  "�  "��Ҹ�"
	 �ԙ���"P*Lgradient_tape/sequential/densenet169/conv5_block12_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block12_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "��Ԫ�"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block12_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}�����޳"��" *$$1""8�
	 �����"
  "  "�  "x����"
	 ������"
  "  "�  "v����"
	 �悇��"
  "  "�  "v؆��"
	 �ꂇ��"
  "  "�  "����"
	 ��Ҭ��"Y*Ugradient_tape/sequential/densenet169/conv5_block12_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block12_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 �悇��"Y*Ugradient_tape/sequential/densenet169/conv5_block12_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2252]"�  ".������Ȣ�"��" *$$1""8�
	 �悇��"Y*Ugradient_tape/sequential/densenet169/conv5_block12_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ܔ��"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block12_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block12_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block12_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block12_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block12_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S��Η����"��" *$$1""8�
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block12_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��򆠏"
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block12_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�Ђ���"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block12_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x蛘��"
	 �ԙ���"
  "  "�  "-zت������"��" *$$1""8�
	 ������"
  "  "�  "-zؼժ����"��" *$$1""8�
	 ������"
  "  "�  "-zȯگ����"��" *$$1""8�
	 ��Ҭ��"
  "  "�  "-|�������"��" *$$1""8�
	 �����"
  "  "�  "������"
	 �����"P*Lgradient_tape/sequential/densenet169/conv5_block12_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 �Ʈ���"P*Lgradient_tape/sequential/densenet169/conv5_block12_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[992]"�  "�Ȅ���"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block12_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[992]"�  "-}Њ���ะ"��" *$$1""8�
	 �����"
  "  "�  "x豇��"
	 �Ĳ���"
  "  "�  "vȠ���"
	 ��ƚ��"
  "  "�  "v�����"
	 ��ƚ��"
  "  "�  "-z��Ï��"��" *$$1""8�
	 �Ʈ���"
  "  "�  "-zж�ȏ�Ȱ"��" *$$1""8�
	 ������"
  "  "�  ".��СΏ���"��" *$$1""8�
	 ������"
  "  "�  "����ҏ"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block11_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~���ӏо�"��" *$$1""8�
	 �Ĳ���"C*?gradient_tape/sequential/densenet169/conv5_block11_concat/Slice"
 �������"
*output" "*
	 �����"
  "  "�  "��Ώ܏"
	 �ȁ���"Y*Ugradient_tape/sequential/densenet169/conv5_block11_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "����ݏ"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block11_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block11_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-�������"��" *$$1""8�
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block11_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block11_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block11_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block11_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R�������"��" *$$1""8�
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block11_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block11_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-Sر������"��" *$$1""8�
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block11_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block11_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block11_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 ������"
  "  "�  "-z��Ǌ����"��" *$$1""8�
	 �ȁ���"
  "  "�  "-|Ъ����ʝ"��" *$$1""8�
	 ������"
  "  "�  "��񏔐"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block11_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block11_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block11_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}����؈�"��" *$$1""8�
	 �����"
  "  "�  "x�Ġ��"
	 ������"
  "  "�  "v�����"
	 �����"
  "  "�  "v�񨝐"
	 �����"
  "  "�  "���ߞ�"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block11_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block11_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 �悇��"Y*Ugradient_tape/sequential/densenet169/conv5_block11_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2252]"�  ".�������޻"��" *$$1""8�
	 �悇��"Y*Ugradient_tape/sequential/densenet169/conv5_block11_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ﶷ�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block11_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��˴��"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block11_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block11_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �ꁩ��"X*Tgradient_tape/sequential/densenet169/conv5_block11_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block11_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S���Ő���"��" *$$1""8�
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block11_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����ΐ"
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block11_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����ΐ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block11_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x���ϐ"
	 ������"
  "  "�  "-z���Ӑ���"��" *$$1""8�
	 ������"
  "  "�  "-z���ؐ���"��" *$$1""8�
	 ������"
  "  "�  "-z���ݐȞ�"��" *$$1""8�
	 �����"
  "  "�  "-|�������"��" *$$1""8�
	 ������"
  "  "�  "�����"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv5_block11_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block11_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[960]"�  "�����"
	 �Ʈ���"P*Lgradient_tape/sequential/densenet169/conv5_block11_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[960]"�  "-}�������"��" *$$1""8�
	 �ꁩ��"
  "  "�  "x����"
	 ��Ț��"
  "  "�  "v����"
	 ���"
  "  "�  "v���"
	 ������"
  "  "�  "-z�ئ��ף"��" *$$1""8�
	 ������"
  "  "�  "-zȭ����À"��" *$$1""8�
	 �Ʈ���"
  "  "�  ".��������"��" *$$1""8�
	 �Ĳ���"
  "  "�  "������"
	 ������"E*Agradient_tape/sequential/densenet169/conv5_block10_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~��Ã����"��" *$$1""8�
	 �ꁩ��"C*?gradient_tape/sequential/densenet169/conv5_block10_concat/Slice"
 �������"
*output" "*
	 ��Ҩ��"
  "  "�  "������"
	 �ȁ���"Y*Ugradient_tape/sequential/densenet169/conv5_block10_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "���"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block10_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "���ꐑ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block10_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-��������"��" *$$1""8�
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block10_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block10_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��� �"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block10_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ��٨��"X*Tgradient_tape/sequential/densenet169/conv5_block10_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R��梑�Տ"��" *$$1""8�
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block10_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block10_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S������"��" *$$1""8�
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block10_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��º��"
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block10_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���絑"
	 ��٨��"X*Tgradient_tape/sequential/densenet169/conv5_block10_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��˶�"
	 ������"
  "  "�  "-zФ������"��" *$$1""8�
	 �ȁ���"
  "  "�  "-|������"��" *$$1""8�
	 ������"
  "  "�  "����Ñ"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block10_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block10_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "����đ"
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block10_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}�бǑ���"��" *$$1""8�
	 �����"
  "  "�  "x���ˑ"
	 ������"
  "  "�  "v��ˑ"
	 �����"
  "  "�  "v��̑"
	 �����"
  "  "�  "��ߟ͑"

	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv5_block10_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block10_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block10_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2252]"�  ".����ב�ͱ"��" *$$1""8�
	 �����"Y*Ugradient_tape/sequential/densenet169/conv5_block10_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"

	 ������"Y*Ugradient_tape/sequential/densenet169/conv5_block10_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
}��?" ����" ��-" ��;"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block10_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block10_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block10_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block10_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S褄�����"��" *$$1""8�
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block10_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block10_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block10_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 ������"
  "  "�  "-z�錂����"��" *$$1""8�
	 ������"
  "  "�  "-zЗχ����"��" *$$1""8�
	 ������"
  "  "�  "-z��������"��" *$$1""8�
	 ��Ҩ��"
  "  "�  "-|�򨐒���"��" *$$1""8�"�  "x�����"
	 ����"
  "  "�  "��ْ��"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv5_block10_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv5_block10_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[928]"�  "��ي��"
	 �Ʈ���"P*Lgradient_tape/sequential/densenet169/conv5_block10_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[928]"�  "-}������ʪ"��" *$$1""8�"�  "x��ќ�"
	 ������"
  "  "�  "x�ē��"
	 ��ߙ��"
  "  "�  "v�����"
	 ������"
  "  "�  "v��䝒"
	 �Й���"
  "  "�  "-z؛㠒�̟"��" *$$1""8�"�  "v��أ�"
	 ������"
  "  "�  "-z�È�����"��" *$$1""8�"�  "v�濨�"
	 �Ʈ���"
  "  "�  ".��������"��" *$$1""8�"�  "x��ʬ�"
	 �ꁩ��"
  "  "�  "����"
	 ������"D*@gradient_tape/sequential/densenet169/conv5_block9_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~��������"��" *$$1""8�"�  "���ɲ�"
	 ��ߙ��"B*>gradient_tape/sequential/densenet169/conv5_block9_concat/Slice"
 �������"
*output" "*
	 ��Ҩ��"
  "  "�  "�؄���"
	 �ȁ���"X*Tgradient_tape/sequential/densenet169/conv5_block9_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "���ڹ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block9_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "���̼�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block9_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-�����"��" *$$1""8�"�  "-R�͂ǒ���"��" *$$1""8�"�  "�Х�ɒ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block9_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���ʒ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block9_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��̒"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block9_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �̑���"W*Sgradient_tape/sequential/densenet169/conv5_block9_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R���Β؃�"��" *$$1""8�"�  "���ђ"
��?" ����" ��" ��"
	 �̚���"W*Sgradient_tape/sequential/densenet169/conv5_block9_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
��?" ����" �" �"
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block9_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�ȶؒ���"��" *$$1""8�"�  ".����ے���"��" *$$1""8�"�  ".����ݒ���"��" *$$1""8�"�  "������"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block9_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block9_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����"
	 �̑���"W*Sgradient_tape/sequential/densenet169/conv5_block9_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "xȞ��"
	 ������"
  "  "�  "-zȟ�����"��" *$$1""8�"�  "x����"
	 �ȁ���"
  "  "�  "-|������"��" *$$1""8�"�  "x�ނ�"
	 ��ٙ��"
  "  "�  "��·�"
	 ��ٙ��"O*Kgradient_tape/sequential/densenet169/conv5_block9_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block9_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "�����"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block9_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}�������"��" *$$1""8�"�  "x�����"
	 �̚���"
  "  "�  "xЎ���"
	 ��ә��"
  "  "�  "v�����"
	 �����"
  "  "�  "v�����"
	 �����"
  "  "�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block9_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block9_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block9_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2244]"�  ".��ܹ�����"��" *$$1""8�"�  ".��Ѯ�����"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".��囋����"��" *$$1""8�"�  "-R�ٰ�����"��" *$$1""8�"�  "���ϐ�"
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block9_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ٟ��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block9_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block9_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block9_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block9_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block9_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�龟��͗"��" *$$1""8�"�  ".�ЃϢ����"��" *$$1""8�"�  ".���椓���"��" *$$1""8�"�  "��Ҟ��"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block9_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���⧓"
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block9_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��񁨓"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block9_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��﨓"
	 ��ٙ��"
  "  "�  "-z��������"��" *$$1""8�"�  "v�꘯�"
	 ������"
  "  "�  "-z��ӱ����"��" *$$1""8�"�  "v��ٴ�"
	 ������"
  "  "�  "-zо���Ȓ�"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "-|��������"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "�����"
	 ��Ҩ��"O*Kgradient_tape/sequential/densenet169/conv5_block9_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block9_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[896]"�  "������"
	 �Ʈ���"O*Kgradient_tape/sequential/densenet169/conv5_block9_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[896]"�  "-}М�ē���"��" *$$1""8�"�  "x���ȓ"
	 ������"
  "  "�  "x���ȓ"
	 ������"
  "  "�  "v���ɓ"
	 �䈖��"
  "  "�  "v���ʓ"
	 ������"
  "  "�  "-z�ރ͓近"��" *$$1""8�"�  "v���ϓ"
	 ������"
  "  "�  "-z���ғ���"��" *$$1""8�"�  "v���ԓ"
	 �Ʈ���"
  "  "�  ".����Փ�Љ"��" *$$1""8�"�  "x���ؓ"
	 ��ߙ��"
  "  "�  "����ړ"
	 ������"D*@gradient_tape/sequential/densenet169/conv5_block8_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~��ۓ��"��" *$$1""8�"�  "����ޓ"
	 ������"B*>gradient_tape/sequential/densenet169/conv5_block8_concat/Slice"
 �������"
*output" "*
	 ��Ҩ��"
  "  "�  "�����"
	 �ȁ���"X*Tgradient_tape/sequential/densenet169/conv5_block8_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�����"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block8_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "��Ѕ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block8_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-�������"��" *$$1""8�"�  "-R؅�����"��" *$$1""8�"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block8_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block8_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ޡ��"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block8_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ��٨��"W*Sgradient_tape/sequential/densenet169/conv5_block8_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R��������"��" *$$1""8�"�  "��߾��"
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block8_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block8_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S��ڄ����"��" *$$1""8�"�  ".�؜�Ȝ�"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  "���Ȍ�"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block8_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ȓ��"
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block8_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���č�"
	 ��٨��"W*Sgradient_tape/sequential/densenet169/conv5_block8_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 ������"
  "  "�  "-z��̑��"��" *$$1""8�"�  "xжߔ�"
	 �ȁ���"
  "  "�  "-|������ϑ"��" *$$1""8�"�  "x�����"
�H��?" ����" ��" ��"
	 ������"
  "  "�  "������"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block8_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block8_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "������"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block8_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}�δ�����"��" *$$1""8�"�  "x�����"
�H��?" ����" ��" ��"
	 �����"
  "  "�  "x��ܢ�"
	 �����"
  "  "�  "v�����"
	 ��ۅ��"
  "  "�  "vؐ���"
	 ��ۅ��"
  "  "�  "�؆̤�"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block8_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ����"X*Tgradient_tape/sequential/densenet169/conv5_block8_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block8_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2244]"�  ".���������"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".��ε�����"��" *$$1""8�"�  ".���ɷ�軿"��" *$$1""8�"�  "-R�׺����"��" *$$1""8�"�  "��䁽�"
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block8_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���۽�"
	 ����"X*Tgradient_tape/sequential/densenet169/conv5_block8_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�гܿ�"
	 ����"W*Sgradient_tape/sequential/densenet169/conv5_block8_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block8_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block8_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block8_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S���̔趔"��" *$$1""8�"�  ".����ϔ���"��" *$$1""8�"�  ".����єП�"��" *$$1""8�"�  "���Ӕ"
	 ����"W*Sgradient_tape/sequential/densenet169/conv5_block8_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����Ԕ"
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block8_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����Ԕ"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block8_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x���Ք"
	 ������"
  "  "�  "-z���ٔ���"��" *$$1""8�"�  "v���ܔ"
	 ������"
  "  "�  "-z���ޔ���"��" *$$1""8�"�  "v����"
	 ������"
  "  "�  "-z�������"��" *$$1""8�"�  "x�چ�"
	 ��Ҩ��"
  "  "�  "-|�����ז"��" *$$1""8�"�  "xЉ��"
	 ��Ę��"
  "  "�  "�����"
	 ��Ҩ��"O*Kgradient_tape/sequential/densenet169/conv5_block8_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block8_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[864]"�  "�����"
	 �Ʈ���"O*Kgradient_tape/sequential/densenet169/conv5_block8_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[864]"�  "-}����ؔ�"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "x�����"
	 �ܚ���"
  "  "�  "v�����"
	 ������"
  "  "�  "vȵ���"
	 �Ȉ���"
  "  "�  "-z��������"��" *$$1""8�"�  "v�ȳ��"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "v؁���"
	 �Ʈ���"
  "  "�  ".���Á����"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "������"
	 ������"D*@gradient_tape/sequential/densenet169/conv5_block7_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~��̆����"��" *$$1""8�"�  "�в���"
	 ������"B*>gradient_tape/sequential/densenet169/conv5_block7_concat/Slice"
 �������"
*output" "*
	 ��Ҩ��"
  "  "�  "�О���"
	 �ȁ���"X*Tgradient_tape/sequential/densenet169/conv5_block7_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "���Ȑ�"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block7_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�𲻓�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block7_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-��������"��" *$$1""8�"�  "-R������"��" *$$1""8�"�  "��矠�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block7_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block7_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block7_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ��٨��"W*Sgradient_tape/sequential/densenet169/conv5_block7_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R�����ؐ�"��" *$$1""8�"�  "�ȴᨕ"
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block7_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block7_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S��а��ݡ"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".�������߻"��" *$$1""8�"�  "�ȡḕ"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block7_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�ส��"
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block7_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���ع�"
	 ��٨��"W*Sgradient_tape/sequential/densenet169/conv5_block7_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�ܵ��"
	 ������"
  "  "�  "-z������"��" *$$1""8�"�  "x����"
	 �ȁ���"
  "  "�  "-|������"��" *$$1""8�"�  "x���ŕ"
	 ������"
  "  "�  "����ƕ"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block7_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block7_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "��ϼǕ"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block7_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}�ϻʕȺ�"��" *$$1""8�"�  "xȪ�Ε"
	 �����"
  "  "�  "x���Ε"
	 �Ԍ���"
  "  "�  "v���Ε"
	 ��ۅ��"
  "  "�  "v�Îϕ"
	 ��ۅ��"
  "  "�  "����Е"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block7_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block7_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 ��ۅ��"X*Tgradient_tape/sequential/densenet169/conv5_block7_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2244]"�  ".����ڕ���"��" *$$1""8�"�  ".����ޕ���"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".�����؞�"��" *$$1""8�"�  "-R�̶����"��" *$$1""8�"�  "�����"
	 ��ۅ��"X*Tgradient_tape/sequential/densenet169/conv5_block7_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block7_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block7_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block7_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block7_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��ۅ��"W*Sgradient_tape/sequential/densenet169/conv5_block7_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�����ތ"��" *$$1""8�"�  ".�������ظ"��" *$$1""8�"�  ".��Ӂ����"��" *$$1""8�"�  "��Ѻ��"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block7_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ��ۅ��"W*Sgradient_tape/sequential/densenet169/conv5_block7_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��򟀖"
dv��?" ����" ��" ��"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block7_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 ������"
  "  "�  "-z�̿����"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-zؼ������"��" *$$1""8�"�  "v𳚍�"
	 ������"
  "  "�  "-z�폖���"��" *$$1""8�"�  "x�򲒖"
	 ��Ҩ��"
  "  "�  "-|�������"��" *$$1""8�"�  "x��Ԗ�"
	 �����"
  "  "�  "���헖"
	 ��Ҩ��"O*Kgradient_tape/sequential/densenet169/conv5_block7_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 �����"O*Kgradient_tape/sequential/densenet169/conv5_block7_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[832]"�  "���"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block7_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[832]"�  "-}��盖���"��" *$$1""8�"�  "x�ό��"
	 ������"
  "  "�  "x�� �"
	 ������"
  "  "�  "v��㠖"
	 �܇���"
  "  "�  "v𩋡�"
	 ������"
  "  "�  "-z������Ԋ"��" *$$1""8�"�  "v��首"
	 �����"
  "  "�  "-zȒ������"��" *$$1""8�"�  "v�ë�"
	 ������"
  "  "�  ".�ȧ������"��" *$$1""8�"�  "x��˯�"
	 ������"
  "  "�  "���"
	 ������"D*@gradient_tape/sequential/densenet169/conv5_block6_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~������Ԋ"��" *$$1""8�"�  "������"
	 ������"B*>gradient_tape/sequential/densenet169/conv5_block6_concat/Slice"
 �������"
*output" "*
	 ��Ҩ��"
  "  "�  "��՗��"
	 �ȁ���"X*Tgradient_tape/sequential/densenet169/conv5_block6_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "����"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block6_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�ؒ徖"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block6_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-�جĖ���"��" *$$1""8�"�  "-Rؽ�ɖ���"��" *$$1""8�"�  "����˖"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block6_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�ȼ�̖"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block6_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�Ȱ�Ζ"
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block6_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block6_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R���Ж�Ӎ"��" *$$1""8�"�  "����Ӗ"
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block6_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��ۅ��"W*Sgradient_tape/sequential/densenet169/conv5_block6_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S���ږ���"��" *$$1""8�"�  ".����ޖ���"��" *$$1""8�"�  ".�������ξ"��" *$$1""8�"�  "��ٕ�"
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block6_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����"
	 ��ۅ��"W*Sgradient_tape/sequential/densenet169/conv5_block6_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��܌�"
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block6_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x����"
	 ������"
  "  "�  "-z�Ӫ���"��" *$$1""8�"�  "x����"
	 �ȁ���"
  "  "�  "-|������"��" *$$1""8�"�  "x����"
	 ������"
  "  "�  "����"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block6_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block6_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "�����"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block6_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}��������"��" *$$1""8�"�  "x�����"
	 �����"
  "  "�  "x���"
�����?" ����" ��" ��"
	 ������"
  "  "�  "v����"
�����?" ����" �" �"
	 ��ۅ��"
  "  "�  "v�����"
�����?" ����" �" �"
	 ��ۅ��"
  "  "�  "��ޙ��"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block6_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block6_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block6_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2244]"�  ".���ن����"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".��ֺ���ۿ"��" *$$1""8�"�  ".���Ў����"��" *$$1""8�"�  "-R��֑����"��" *$$1""8�"�  "���ǔ�"
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block6_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��閕�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block6_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block6_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block6_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block6_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block6_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�����"��" *$$1""8�"�  ".������з�"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  "���٫�"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block6_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ԡ��"
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block6_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block6_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�Ӯ��"
	 ������"
  "  "�  "-z�퇱����"��" *$$1""8�"�  "v��봗"
	 ������"
  "  "�  "-z؇˷�ض�"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-z��ռ����"��" *$$1""8�"�  "x�����"
	 �����"
  "  "�  "-|������"��" *$$1""8�"�  "x���×"
	 �쇗��"
  "  "�  "����ė"
	 ��Ҩ��"O*Kgradient_tape/sequential/densenet169/conv5_block6_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ��ۅ��"O*Kgradient_tape/sequential/densenet169/conv5_block6_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[800]"�  "����ŗ"
	 �����"O*Kgradient_tape/sequential/densenet169/conv5_block6_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[800]"�  "-}���ȗȟ�"��" *$$1""8�"�  "x���̗"
	 ������"
  "  "�  "x���̗"
	 ������"
  "  "�  "v���͗"
	 ������"
  "  "�  "v��͗"
	 ������"
  "  "�  "-z�͕З���"��" *$$1""8�"�  "v���ӗ"
	 ��ۅ��"
  "  "�  "-z���֗���"��" *$$1""8�"�  "v���ؗ"
	 �����"
  "  "�  ".��ԇڗ���"��" *$$1""8�"�  "x���ܗ"
�Y2��?" ����" ��'" ��'"
	 ������"
  "  "�  "��ǂޗ"
	 ������"D*@gradient_tape/sequential/densenet169/conv5_block5_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~Ћ�ߗ�˕"��" *$$1""8�"�  "�����"
	 ������"B*>gradient_tape/sequential/densenet169/conv5_block5_concat/Slice"
 �������"
*output" "*
�Y2��?" ����" ��'" ��'"
	 ��Ҩ��"
  "  "�  "�З��"
	 �ȁ���"X*Tgradient_tape/sequential/densenet169/conv5_block5_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�����"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block5_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block5_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-�������"��" *$$1""8�"�  "-R��������"��" *$$1""8�"�  "�Б���"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block5_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block5_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block5_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ��٨��"W*Sgradient_tape/sequential/densenet169/conv5_block5_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R�������"��" *$$1""8�"�  "��Â�"
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block5_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block5_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S��托ж�"��" *$$1""8�"�  ".��玎����"��" *$$1""8�"�  ".��˽�����"��" *$$1""8�"�  "������"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block5_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ʤ��"
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block5_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���疘"
	 ��٨��"W*Sgradient_tape/sequential/densenet169/conv5_block5_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "x�Р�"
	 �ȁ���"
  "  "�  "-|Ȥ�����"��" *$$1""8�"�  "x�ԧ��"
	 ��ږ��"
  "  "�  "������"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block5_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block5_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "�క��"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block5_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}��ͮ�ب�"��" *$$1""8�"�  "x�����"
	 �����"
  "  "�  "x؀���"
	 ��Ԗ��"
  "  "�  "v�꧵�"
���?" ����" �" �"
	 ��؄��"
  "  "�  "v��ٵ�"
���?" ����" �" �"
	 ��؄��"
  "  "�  "��۷�"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block5_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ����"X*Tgradient_tape/sequential/densenet169/conv5_block5_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block5_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2236]"�  ".��ūĘ��"��" *$$1""8�"�  ".����Ș���"��" *$$1""8�"�  ".����ʘ���"��" *$$1""8�"�  ".����͘��"��" *$$1""8�"�  "-R���И���"��" *$$1""8�"�  "����Ҙ"
	 �悇��"X*Tgradient_tape/sequential/densenet169/conv5_block5_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ͷӘ"
	 ����"X*Tgradient_tape/sequential/densenet169/conv5_block5_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����՘"
	 ����"W*Sgradient_tape/sequential/densenet169/conv5_block5_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ��ũ��"W*Sgradient_tape/sequential/densenet169/conv5_block5_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block5_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block5_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�������"��" *$$1""8�"�  ".�������"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  "������"
	 ����"W*Sgradient_tape/sequential/densenet169/conv5_block5_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block5_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ��ũ��"W*Sgradient_tape/sequential/densenet169/conv5_block5_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�ݽ��"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-z��Ċ����"��" *$$1""8�"�  "x�����"
	 ��Ҩ��"
  "  "�  "-|Џю���"��" *$$1""8�"�  "x��Ƒ�"
	 �䮖��"
  "  "�  "���뒙"
	 ��Ҩ��"O*Kgradient_tape/sequential/densenet169/conv5_block5_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 �悇��"O*Kgradient_tape/sequential/densenet169/conv5_block5_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[768]"�  "���ꓙ"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block5_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[768]"�  "-}�޳����"��" *$$1""8�"�  "x�ě��"
	 ������"
  "  "�  "x��䛙"
	 ������"
  "  "�  "vؙ���"
	 ������"
  "  "�  "v�����"
	 ������"
  "  "�  "-z��ϟ�࿄"��" *$$1""8�"�  "vࡩ��"
	 �悇��"
  "  "�  "-zЇ夙���"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  ".��娙���"��" *$$1""8�"�  "x�镬�"
	 ������"
  "  "�  "���έ�"
	 ������"D*@gradient_tape/sequential/densenet169/conv5_block4_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~������"��" *$$1""8�"�  "������"
	 ������"B*>gradient_tape/sequential/densenet169/conv5_block4_concat/Slice"
 �������"
*output" "*
	 ��Ҩ��"
  "  "�  "��і��"
	 �ȁ���"X*Tgradient_tape/sequential/densenet169/conv5_block4_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "������"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block4_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "��ر��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block4_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-������"��" *$$1""8�"�  "-R���Ǚ���"��" *$$1""8�"�  "���ə"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block4_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ѫʙ"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block4_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����̙"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block4_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ��٨��"W*Sgradient_tape/sequential/densenet169/conv5_block4_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R���Ι��"��" *$$1""8�"�  "��߻љ"
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block4_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block4_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�זٙ�מ"��" *$$1""8�"�  ".����ܙ���"��" *$$1""8�"�  ".���ޙଷ"��" *$$1""8�"�  "�����"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block4_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����"
	 �悇��"W*Sgradient_tape/sequential/densenet169/conv5_block4_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����"
	 ��٨��"W*Sgradient_tape/sequential/densenet169/conv5_block4_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x����"
	 ������"
  "  "�  "-z������"��" *$$1""8�"�  "x�Ϫ�"
	 �ȁ���"
  "  "�  "-|����ط�"��" *$$1""8�"�  "x����"
	 ������"
  "  "�  "�����"
K<�l��?" ����" ��" ��"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block4_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
K<�l��?" ����" �" �"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block4_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "����"
K<�l��?" ����" �" �"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block4_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}����٥"��" *$$1""8�"�  "x�����"
	 �����"
  "  "�  "x�����"
	 ������"
  "  "�  "v�����"
	 ��؄��"
  "  "�  "v�����"
	 ��؄��"
  "  "�  "������"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block4_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block4_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv5_block4_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2236]"�  ".���������"��" *$$1""8�"�  ".�خ������"��" *$$1""8�"�  ".������؂�"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  "-R诎�����"��" *$$1""8�"�  "������"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv5_block4_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block4_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block4_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block4_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block4_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block4_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S𥕠��ڐ"��" *$$1""8�"�  ".�������؁"��" *$$1""8�"�  ".���Ц����"��" *$$1""8�"�  "������"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block4_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�Л�"
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block4_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��䞪�"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block4_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x����"
	 ������"
  "  "�  "-z蚵��ๆ"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-z��Ƴ����"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-z�鱸���"��" *$$1""8�"�  "x��ߺ�"
	 ��Ҩ��"
  "  "�  "-|ȧ������"��" *$$1""8�"�  "x�߅��"
	 ��ו��"
  "  "�  "������"
	 ��Ҩ��"O*Kgradient_tape/sequential/densenet169/conv5_block4_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 �悇��"O*Kgradient_tape/sequential/densenet169/conv5_block4_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[736]"�  "�ȝ���"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block4_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[736]"�  "-}���Ě��"��" *$$1""8�"�  "x���Ț"
	 ������"
  "  "�  "x���Ț"
	 ������"
  "  "�  "vм�ɚ"
	 ������"
  "  "�  "v�ҷɚ"
	 ������"
  "  "�  "-z���̚���"��" *$$1""8�"�  "v���Κ"
	 �悇��"
  "  "�  "-z���њ���"��" *$$1""8�"�  "v���Ӛ"
	 ������"
  "  "�  ".��Ȍ՚���"��" *$$1""8�"�  "x���ך"
	 ������"
  "  "�  "����ؚ"
	 ������"D*@gradient_tape/sequential/densenet169/conv5_block3_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~��ښ���"��" *$$1""8�"�  "����ޚ"
	 ������"B*>gradient_tape/sequential/densenet169/conv5_block3_concat/Slice"
 �������"
*output" "*
	 ��Ҩ��"
  "  "�  "��Δ�"
	 �ȁ���"X*Tgradient_tape/sequential/densenet169/conv5_block3_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�����"
	 ��Օ��"X*Tgradient_tape/sequential/densenet169/conv5_block3_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block3_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-������"��" *$$1""8�"�  "-R�������"��" *$$1""8�"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block3_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��Է��"
	 ��Օ��"X*Tgradient_tape/sequential/densenet169/conv5_block3_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ܹ��"
	 ��Օ��"W*Sgradient_tape/sequential/densenet169/conv5_block3_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ��ە��"W*Sgradient_tape/sequential/densenet169/conv5_block3_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R������Љ"��" *$$1""8�"�  "������"
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block3_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block3_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�����"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  "���ꋛ"
	 ��Օ��"W*Sgradient_tape/sequential/densenet169/conv5_block3_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��̶��"
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block3_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���⌛"
	 ��ە��"W*Sgradient_tape/sequential/densenet169/conv5_block3_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��ȍ�"
	 ������"
  "  "�  "-z�ҝ�����"��" *$$1""8�"�  "x臭��"
	 �ȁ���"
  "  "�  "-|�������"��" *$$1""8�"�  "x��՘�"
	 ������"
  "  "�  "���ٙ�"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block3_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block3_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "���ǚ�"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block3_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}��ם����"��" *$$1""8�"�  "x�����"
	 �����"
  "  "�  "x�����"
	 �즕��"
  "  "�  "v��Ӣ�"
	 �����"
  "  "�  "vओ��"
	 ��؄��"
  "  "�  "������"
	 ��Օ��"X*Tgradient_tape/sequential/densenet169/conv5_block3_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block3_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv5_block3_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2236]"�  ".��߅�����"��" *$$1""8�"�  ".��ğ�����"��" *$$1""8�"�  ".���ش����"��" *$$1""8�"�  ".���綛���"��" *$$1""8�"�  "-R��깛���"��" *$$1""8�"�  "�����"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv5_block3_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���ȼ�"
	 �����"X*Tgradient_tape/sequential/densenet169/conv5_block3_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���Ͼ�"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block3_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block3_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block3_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block3_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S�ؾʛ���"��" *$$1""8�"�  ".����Λ���"��" *$$1""8�"�  ".�ȉ�Л�˿"��" *$$1""8�"�  "��ȣӛ"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block3_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�ذ�ӛ"
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block3_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�ȅ�ԛ"
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block3_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x���ԛ"
	 ������"
  "  "�  "-z���؛М�"��" *$$1""8�"�  "v���ܛ"
	 ������"
  "  "�  "-zؿ�ޛ���"��" *$$1""8�"�  "v����"
	 ������"
  "  "�  "-z�������"��" *$$1""8�"�  "x����"
	 ��Օ��"
  "  "�  "-|Ƞ�����"��" *$$1""8�"�  "x����"
	 ������"
  "  "�  "�����"
	 ��Ҩ��"O*Kgradient_tape/sequential/densenet169/conv5_block3_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ��؄��"O*Kgradient_tape/sequential/densenet169/conv5_block3_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[704]"�  "����"
	 �悇��"O*Kgradient_tape/sequential/densenet169/conv5_block3_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[704]"�  "-}�������"��" *$$1""8�"�  "x��"
	 ������"
  "  "�  "x����"
	 �����"
  "  "�  "vؖ��"
	 �����"
  "  "�  "v�����"
	 �����"
  "  "�  "-z������Ԓ"��" *$$1""8�"�  "v؞���"
	 ��؄��"
  "  "�  "-z�������"��" *$$1""8�"�  "v�����"
	 �悇��"
  "  "�  ".���������"��" *$$1""8�"�  "x��Ղ�"
	 ������"
  "  "�  "�𤄄�"
	 ������"D*@gradient_tape/sequential/densenet169/conv5_block2_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~�˖�����"��" *$$1""8�"�  "���ڈ�"
	 ������"B*>gradient_tape/sequential/densenet169/conv5_block2_concat/Slice"
 �������"
*output" "*
	 ��Ҩ��"
  "  "�  "������"
	 �ȁ���"X*Tgradient_tape/sequential/densenet169/conv5_block2_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "������"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block2_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block2_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-�Ѭ���Ϩ"��" *$$1""8�"�  "-R��霜���"��" *$$1""8�"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block2_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block2_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block2_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ��٨��"W*Sgradient_tape/sequential/densenet169/conv5_block2_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R��������"��" *$$1""8�"�  "������"
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block2_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block2_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S��������"��" *$$1""8�"�  ".�ؐ������"��" *$$1""8�"�  ".���ɳ����"��" *$$1""8�"�  "��Ǆ��"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block2_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���Ͷ�"
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block2_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ��٨��"W*Sgradient_tape/sequential/densenet169/conv5_block2_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��۷�"
	 ������"
  "  "�  "-z������Ǚ"��" *$$1""8�"�  "x�����"
	 �ȁ���"
  "  "�  "-|Ȩ޿����"��" *$$1""8�"�  "x���"
	 ��۔��"
  "  "�  "���Ü"
	 �����"O*Kgradient_tape/sequential/densenet169/conv5_block2_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block2_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "����Ĝ"
	 �����"O*Kgradient_tape/sequential/densenet169/conv5_block2_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}���ǜ��"��" *$$1""8�"�  "xЫ�˜"
	 �����"
  "  "�  "x؛�̜"
	 ��Ք��"
  "  "�  "v���̜"
	 ������"
  "  "�  "v���͜"
	 ������"
  "  "�  "����Ϝ"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv5_block2_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block2_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv5_block2_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2236]"�  ".����ۜ���"��" *$$1""8�"�  ".���ߜ�ѓ"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".��ҷ����"��" *$$1""8�"�  "-R�������"��" *$$1""8�"�  "�����"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv5_block2_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block2_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block2_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block2_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block2_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block2_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S������ؽ"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".��Ѯ�����"��" *$$1""8�"�  "������"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block2_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���Å�"
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block2_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block2_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��݆�"
	 �����"
  "  "�  "-z�ƾ����"��" *$$1""8�"�  "vК֍�"
	 ������"
  "  "�  "-z������ٿ"��" *$$1""8�"�  "v�����"
	 �����"
  "  "�  "-zత�����"��" *$$1""8�"�  "x����"
	 ��Ҩ��"
  "  "�  "-|��՚���"��" *$$1""8�"�  "x੻��"
	 ������"
  "  "�  "������"
	 ��Ҩ��"O*Kgradient_tape/sequential/densenet169/conv5_block2_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ��؄��"O*Kgradient_tape/sequential/densenet169/conv5_block2_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[672]"�  "������"
	 �悇��"O*Kgradient_tape/sequential/densenet169/conv5_block2_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[672]"�  "-}خ������"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "x�����"
	 ��ޓ��"
  "  "�  "v�����"
	 ������"
  "  "�  "v��©�"
	 �����"
  "  "�  "-z��۬����"��" *$$1""8�"�  "v��ܯ�"
	 ��؄��"
  "  "�  "-z�薲����"��" *$$1""8�"�  "v��ٴ�"
	 �悇��"
  "  "�  ".���������"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "������"
	 ������"D*@gradient_tape/sequential/densenet169/conv5_block1_concat/Slice_1"
 �������"
*output" "*[200,32,1,1]"�  "-~ض������"��" *$$1""8�"�  "��Ň��"
	 ��ޓ��"B*>gradient_tape/sequential/densenet169/conv5_block1_concat/Slice"
 �������"
*output" "*
	 ��Ҩ��"
  "  "�  "����Ɲ"
	 �ȁ���"X*Tgradient_tape/sequential/densenet169/conv5_block1_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "����ǝ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block1_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "����ʝ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block1_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "	*[288]"�  "-���Н���"��" *$$1""8�"�  "-R��ם؋�"��" *$$1""8�"�  "����ڝ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block1_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�؋�ڝ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block1_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����ݝ"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block1_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block1_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-RЕ�ߝ���"��" *$$1""8�"�  "����"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block1_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block1_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S������"��" *$$1""8�"�  ".�������"��" *$$1""8�"�  ".������蕟"��" *$$1""8�"�  "������"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block1_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block1_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�Ȏ���"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block1_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x،���"
	 ������"
  "  "�  "-z�ϖ��й�"��" *$$1""8�"�  "x辀��"
	 �ȁ���"
  "  "�  "-|��燞���"��" *$$1""8�"�  "x��͊�"
	 ��ؓ��"
  "  "�  "���㋞"
	 ��ؓ��"O*Kgradient_tape/sequential/densenet169/conv5_block1_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block1_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "���݌�"
	 �����"O*Kgradient_tape/sequential/densenet169/conv5_block1_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}������ٲ"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "x��ו�"
	 �줺��"
  "  "�  "v�����"
	 ������"
  "  "�  "v�����"
	 ������"
  "  "�  "���旞"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block1_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block1_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv5_block1_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2228]"�  ".���ã�д�"��" *$$1""8�"�  ".�Ȏ���Ѝ�"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".�謑����"��" *$$1""8�"�  "-R�ۢ�����"��" *$$1""8�"�  "��⇲�"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv5_block1_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�Ȃֲ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv5_block1_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���ഞ"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block1_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block1_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �����"W*Sgradient_tape/sequential/densenet169/conv5_block1_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block1_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2188]"�  "-S���Þ�ݜ"��" *$$1""8�"�  ".����ƞ��"��" *$$1""8�"�  ".����ɞȜ�"��" *$$1""8�"�  "�Ȱ�˞"
	 ������"W*Sgradient_tape/sequential/densenet169/conv5_block1_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ƛ̞"
	 ��؄��"W*Sgradient_tape/sequential/densenet169/conv5_block1_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���̞"
	 ��Ҩ��"W*Sgradient_tape/sequential/densenet169/conv5_block1_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x���͞"
	 ��ؓ��"
  "  "�  "-z���ў���"��" *$$1""8�"�  "vؗ�Ԟ"
	 ������"
  "  "�  "-z���֞���"��" *$$1""8�"�  "v���ٞ"
	 �����"
  "  "�  "-z���ܞ��"��" *$$1""8�"�  "x���ߞ"
	 ������"
  "  "�  "-|�ф����"��" *$$1""8�"�  "xȾ��"
	 ������"
  "  "�  "����"
	 ������"O*Kgradient_tape/sequential/densenet169/conv5_block1_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ��؄��"O*Kgradient_tape/sequential/densenet169/conv5_block1_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[640]"�  "�����"
	 �悇��"O*Kgradient_tape/sequential/densenet169/conv5_block1_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[640]"�  "-}����诽"��" *$$1""8�"�  "x�����"
	 �����"
  "  "�  "x����"
	 ������"
  "  "�  "v����"
	 ������"
  "  "�  "v؛���"
	 ������"
  "  "�  "-z�����࿁"��" *$$1""8�"�  "v�Υ��"
	 ��؄��"
  "  "�  "-z��ą���"��" *$$1""8�"�  "v��눟"
	 �悇��"
  "  "�  ".���������"��" *$$1""8�"�  "x�����"
	 ��ޓ��"
  "  "�  "��ߗ��"
	 ��Ҩ��"G*Cgradient_tape/sequential/densenet169/pool4_pool/AvgPool/AvgPoolGrad"
 �������"
*output" "*
	 ������"G*Cgradient_tape/sequential/densenet169/pool4_pool/AvgPool/AvgPoolGrad"
 �������"*temp" "*
	 ��ǚ��"G*Cgradient_tape/sequential/densenet169/pool4_pool/AvgPool/AvgPoolGrad"
 �������"*temp" "*
	 ��ǚ��"G*Cgradient_tape/sequential/densenet169/pool4_pool/AvgPool/AvgPoolGrad"
 �������"  "�  "��ӭ��"
	 ������"G*Cgradient_tape/sequential/densenet169/pool4_pool/AvgPool/AvgPoolGrad"
 �������"  "�  "x�폟�"
	 ������"
  "  "�  "�؁ʠ�"
	 ��ǚ��"O*Kgradient_tape/sequential/densenet169/pool4_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1280,640]"�  "��ս��"
	 ������"O*Kgradient_tape/sequential/densenet169/pool4_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[640,1280,1,1]"�  "���䦟"
	 ��؄��"O*Kgradient_tape/sequential/densenet169/pool4_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2600]"�  ".��罭����"��" *$$1""8�"�  ".�������ܾ"��" *$$1""8�"�  ".��ת����"��" *$$1""8�"�  ".���ù����"��" *$$1""8�"�  "-R��������"��" *$$1""8�"�  "�ȭ���"
	 ��؄��"O*Kgradient_tape/sequential/densenet169/pool4_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����"
	 ������"O*Kgradient_tape/sequential/densenet169/pool4_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����ş"
	 �ڍ���"N*Jgradient_tape/sequential/densenet169/pool4_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1280,2,2]"�  "����ǟ"
	 ������"N*Jgradient_tape/sequential/densenet169/pool4_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[640,1280,1,1]"�  "-R���ʟ��"��" *$$1""8�"�  "�௎ϟ"
	 ������"N*Jgradient_tape/sequential/densenet169/pool4_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1280,2,2]"�  "����ԟ"
	 ��؄��"N*Jgradient_tape/sequential/densenet169/pool4_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S���ڟ؆�"��" *$$1""8�"�  ".����ޟ�׋"��" *$$1""8�"�  ".�������"��" *$$1""8�"�  "�����"
	 �ڍ���"N*Jgradient_tape/sequential/densenet169/pool4_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���"
	 ��؄��"N*Jgradient_tape/sequential/densenet169/pool4_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����"
	 ������"N*Jgradient_tape/sequential/densenet169/pool4_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "y����"
	 ��Ҩ��"
  "  "�  "-z����س�"��" *$$1""8�"�  "z�Ї�"
	 ��ǚ��"
  "  "�  "-|��������"��" *$$1""8�"�  "z�����"
	 ������"
  "  "�  "������"
	 �ڍ���"F*Bgradient_tape/sequential/densenet169/pool4_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1280,2,2]"�  "������"
	 ������"F*Bgradient_tape/sequential/densenet169/pool4_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1280]"�  "��̙��"
	 �臝��"F*Bgradient_tape/sequential/densenet169/pool4_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1280]"�  "-}��������"��" *$$1""8�"�  "z�ֆ�"
	 ������"
  "  "�  "z�����"
	 ��ŏ��"
  "  "�  "v�����"
	 ������"
  "  "�  "v��쇠"
	 ������"
  "  "�  "������"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block32_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~��������"��" *$$1""8�"�  "��ď�"
	 �ކ���"C*?gradient_tape/sequential/densenet169/conv4_block32_concat/Slice"
 �������"
*output" "*[200,1248,2,2]"�  "-~��������"��" *$$1""8�"�  "z����"
	 �ڍ���"
  "  "�  "-z��������"��" *$$1""8�"�  "vȖ���"
	 ������"
  "  "�  "-zخ�����"��" *$$1""8�"�  "v��"
	 �臝��"
  "  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block32_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "���ͧ�"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block32_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "������"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block32_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block32_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block32_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���à"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R���Š��"��" *$$1""8�"�  "����Ƞ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S���Р���"��" *$$1""8�"�  ".�إ�Ԡ���"��" *$$1""8�"�  ".����֠���"��" *$$1""8�"�  "����٠"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����ڠ"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���ڠ"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block32_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x���ܠ"
	 �줺��"
  "  "�  "-zج����"��" *$$1""8�"�  "x����"
	 ������"
  "  "�  "-|�������"��" *$$1""8�"�  "x�¡�"
	 ������"
  "  "�  "�����"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv4_block32_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block32_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "��Ԫ�"
	 �����"P*Lgradient_tape/sequential/densenet169/conv4_block32_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}������թ"��" *$$1""8�"�  "x�ͧ�"
	 ������"
  "  "�  "x����"
	 ������"
  "  "�  "vЍ��"
	 �����"
  "  "�  "v����"
	 �����"
  "  "�  "�ȱ���"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block32_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1248,128]"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block32_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1248,1,1]"�  "��Ձ��"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block32_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2280]"�  ".�������"��" *$$1""8�"�  ".��բ�����"��" *$$1""8�"�  ".���冡���"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  "-R𴍌����"��" *$$1""8�"�  "������"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block32_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block32_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�ؕ���"
	 �ڍ���"X*Tgradient_tape/sequential/densenet169/conv4_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1248,2,2]"�  "��݌��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1248,1,1]"�  "-R�˔�����"��" *$$1""8�"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1248,2,2]"�  "������"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S��������"��" *$$1""8�"�  ".��є�����"��" *$$1""8�"�  ".�������׻"��" *$$1""8�"�  "���娡"
	 �ڍ���"X*Tgradient_tape/sequential/densenet169/conv4_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�𴳩�"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���⩡"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block32_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 ��Ҩ��"
  "  "�  "-z��������"��" *$$1""8�"�  "vإ���"
	 ������"
  "  "�  "-z��ų����"��" *$$1""8�"�  "vض���"
	 �����"
  "  "�  "-z�겺����"��" *$$1""8�"�  "x�ڡ��"
	 ������"
  "  "�  "-|����ߞ"��" *$$1""8�"�  "z���¡"
	 �̟���"
  "  "�  "����á"
	 �ڍ���"P*Lgradient_tape/sequential/densenet169/conv4_block32_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1248,2,2]"�  "��̰ġ"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block32_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1248]"�  "����ġ"
	 �臝��"P*Lgradient_tape/sequential/densenet169/conv4_block32_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1248]"�  "-}�ʃȡ���"��" *$$1""8�"�  "zȠ�ˡ"
	 ������"
  "  "�  "z�̡"
	 �쫋��"
  "  "�  "v���͡"
	 ������"
  "  "�  "v؅�Ρ"
	 ������"
  "  "�  "-z��ѡج�"��" *$$1""8�"�  "v���ԡ"
	 ������"
  "  "�  "-z���֡���"��" *$$1""8�"�  "v���١"
	 �臝��"
  "  "�  ".����ۡء�"��" *$$1""8�"�  "z���ޡ"
	 �ކ���"
  "  "�  "����ߡ"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block31_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~������"��" *$$1""8�"�  "�����"
	 �ކ���"C*?gradient_tape/sequential/densenet169/conv4_block31_concat/Slice"
 �������"
*output" "*[200,1216,2,2]"�  "-~����؁"��" *$$1""8�"�  "z����"
	 �ڍ���"
  "  "�  "��̻�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block31_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�踬�"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block31_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�����"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block31_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block31_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���Ā�"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block31_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R�ٹ���׸"��" *$$1""8�"�  "���ڈ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S�����"��" *$$1""8�"�  ".���镢���"��" *$$1""8�"�  ".��ؘ���ƾ"��" *$$1""8�"�  "���֚�"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��՛�"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block31_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x౸��"
	 �줺��"
  "  "�  "-z������"��" *$$1""8�"�  "x�ʋ��"
	 ������"
  "  "�  "-|��ߤ����"��" *$$1""8�"�  "x��ŧ�"
	 �쒋��"
  "  "�  "���Ǩ�"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv4_block31_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block31_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "������"
	 �����"P*Lgradient_tape/sequential/densenet169/conv4_block31_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}��������"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "x�᯲�"
	 ������"
  "  "�  "v��Ӳ�"
	 �����"
  "  "�  "v�����"
	 �����"
  "  "�  "��Ʋ��"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block31_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1216,128]"�  "��ň��"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block31_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1216,1,1]"�  "������"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block31_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2280]"�  ".��Ҕ���Ĺ"��" *$$1""8�"�  ".����Ģ���"��" *$$1""8�"�  ".�У�ƢȽ�"��" *$$1""8�"�  ".����ɢ�ƾ"��" *$$1""8�"�  "-R���̢���"��" *$$1""8�"�  "���΢"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block31_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����Ϣ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block31_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����Ѣ"
	 �ڍ���"X*Tgradient_tape/sequential/densenet169/conv4_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1216,2,2]"�  "����Ӣ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1216,1,1]"�  "-R���Ԣ��"��" *$$1""8�"�  "��ͯע"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1216,2,2]"�  "���ۢ"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S蒭ߢȎ�"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".������"��" *$$1""8�"�  "��Γ�"
	 �ڍ���"X*Tgradient_tape/sequential/densenet169/conv4_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ۓ�"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����"
e���?" ����" ��&" ��&"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block31_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x���"
	 ��Ҩ��"
  "  "�  "-zȈ�����"��" *$$1""8�"�  "v�ǔ�"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "v�����"
	 �����"
  "  "�  "-z�ž�����"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "-|Ȃ������"��" *$$1""8�"�  "z�ơ��"
	 ������"
  "  "�  "��򱁣"
	 �ڍ���"P*Lgradient_tape/sequential/densenet169/conv4_block31_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1216,2,2]"�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block31_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1216]"�  "���˃�"
	 �ڄ���"P*Lgradient_tape/sequential/densenet169/conv4_block31_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1216]"�  "-}��ꆣ���"��" *$$1""8�"�  "z��ˊ�"
	 ������"
  "  "�  "z�����"
	 �잇��"
  "  "�  "v��"
	 ��ë��"
  "  "�  "v��ދ�"
	 ��ë��"
  "  "�  "-z�����д�"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "v�����"
	 �ڄ���"
  "  "�  ".���ؘ�Ч�"��" *$$1""8�"�  "z�˒��"
�?" ����" ���" ���"
	 �ކ���"
  "  "�  "���Н�"
�?" ����" ��" ��"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block30_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~Ђ��ף"��" *$$1""8�"�  "���Ң�"
	 �ކ���"C*?gradient_tape/sequential/densenet169/conv4_block30_concat/Slice"
 �������"
*output" "*[200,1184,2,2]"�  "-~��Σ����"��" *$$1""8�"�  "z�����"
�?" ����" ���" ���"
	 �ڍ���"
  "  "�  "��񿧣"
�?" ����" ��	" ��
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block30_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "��ݫ��"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block30_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "���۬�"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block30_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block30_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���Խ�"
�?" ����" ��	" ��	"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block30_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���ٿ�"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R���£���"��" *$$1""8�"�  "�؏�ţ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S��ͣ���"��" *$$1""8�"�  ".����У�и"��" *$$1""8�"�  ".�؁�ӣ���"��" *$$1""8�"�  "��ļգ"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���֣"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�誷֣"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block30_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x���ף"
	 �줺��"
  "  "�  "-z���ۣ�֕"��" *$$1""8�"�  "x����"
	 ������"
  "  "�  "-|�������"��" *$$1""8�"�  "x����"
�?" ����" ��" ��"
	 �셇��"
  "  "�  "��ȉ�"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv4_block30_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block30_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "�����"
	 �����"P*Lgradient_tape/sequential/densenet169/conv4_block30_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}��������"��" *$$1""8�"�  "x����"
�?" ����" ��" ��"
	 ������"
  "  "�  "xب��"
	 �����"
  "  "�  "v����"
	 �����"
  "  "�  "v����"
	 �����"
  "  "�  "��˽��"
�?" ����" ��%" ��%"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block30_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1184,128]"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block30_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1184,1,1]"�  "�എ��"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block30_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2280]"�  ".���������"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".������Џ�"��" *$$1""8�"�  ".�������֔"��" *$$1""8�"�  "-R��������"��" *$$1""8�"�  "����"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block30_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���ڏ�"
�?" ����" ��%" ��%"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block30_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���ᑤ"
	 �ڍ���"X*Tgradient_tape/sequential/densenet169/conv4_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1184,2,2]"�  "���"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1184,1,1]"�  "-R�������"��" *$$1""8�"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1184,2,2]"�  "������"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S��������"��" *$$1""8�"�  ".��ȑ��؂�"��" *$$1""8�"�  ".�Љ�����"��" *$$1""8�"�  "�з㨤"
	 �ڍ���"X*Tgradient_tape/sequential/densenet169/conv4_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ѳ��"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���橤"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block30_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��ƪ�"
	 ��Ҩ��"
  "  "�  "-zȎ���ऍ"��" *$$1""8�"�  "v��"
	 ������"
  "  "�  "-z��ֳ����"��" *$$1""8�"�  "v�����"
	 �����"
  "  "�  "-z��̸�؄�"��" *$$1""8�"�  "x𼉻�"
	 ������"
  "  "�  "-|��������"��" *$$1""8�"�  "z�����"
	 �̅���"
  "  "�  "��Ûä"
	 �ڍ���"P*Lgradient_tape/sequential/densenet169/conv4_block30_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1184,2,2]"�  "���ä"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block30_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1184]"�  "���Ĥ"
��?" ����" �%" �&"
	 �ڄ���"P*Lgradient_tape/sequential/densenet169/conv4_block30_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1184]"�  "-}�ϺǤ���"��" *$$1""8�"�  "z��ˤ"
	 ������"
  "  "�  "z���ˤ"
	 ������"
  "  "�  "v���ˤ"
	 ������"
  "  "�  "v��ͤ"
	 ������"
  "  "�  "-z���Ф���"��" *$$1""8�"�  "v���Ӥ"
	 ������"
  "  "�  "-z��֤���"��" *$$1""8�"�  "v���ؤ"
	 �ڄ���"
  "  "�  ".����ڤ���"��" *$$1""8�"�  "z���ܤ"
	 �ކ���"
  "  "�  "����ޤ"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block29_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~���ߤ���"��" *$$1""8�"�  "�����"
	 ������"C*?gradient_tape/sequential/densenet169/conv4_block29_concat/Slice"
 �������"
*output" "*[200,1152,2,2]"�  "-~�������"��" *$$1""8�"�  "z����"
	 �ڍ���"
  "  "�  "����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block29_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�����"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block29_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "������"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block29_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block29_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�觴��"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block29_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��Ǯ��"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R��ڂ����"��" *$$1""8�"�  "���օ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S���௳"��" *$$1""8�"�  ".��񫒥���"��" *$$1""8�"�  ".���͔����"��" *$$1""8�"�  "��ǃ��"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�ؘї�"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block29_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��ܘ�"
	 �줺��"
  "  "�  "-z��������"��" *$$1""8�"�  "x�㛟�"
	 ������"
  "  "�  "-|��砥�Ǆ"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "��佤�"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv4_block29_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block29_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "���ަ�"
	 �����"P*Lgradient_tape/sequential/densenet169/conv4_block29_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}��������"��" *$$1""8�"�  "x�䟮�"
	 ������"
  "  "�  "x��֮�"
	 �����"
  "  "�  "v����"
	 �����"
  "  "�  "v�����"
	 �����"
  "  "�  "���Ⱕ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block29_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1152,128]"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block29_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1152,1,1]"�  "��꧶�"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block29_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2272]"�  ".�Ƚ�����"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".��å��"��" *$$1""8�"�  ".����ťȽ�"��" *$$1""8�"�  "-R�ϻȥ���"��" *$$1""8�"�  "����ʥ"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block29_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��հ˥"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block29_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����ͥ"
	 �ކ���"X*Tgradient_tape/sequential/densenet169/conv4_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1152,2,2]"�  "����ϥ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1152,1,1]"�  "-R���ϥ���"��" *$$1""8�"�  "����ҥ"
	 �ڍ���"X*Tgradient_tape/sequential/densenet169/conv4_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1152,2,2]"�  "����֥"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S���ڥ���"��" *$$1""8�"�  ".����ݥ���"��" *$$1""8�"�  ".����ߥ�Ŵ"��" *$$1""8�"�  "�����"
	 �ކ���"X*Tgradient_tape/sequential/densenet169/conv4_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block29_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x����"
	 ��Ҩ��"
  "  "�  "-z�����â"��" *$$1""8�"�  "v����"
	 ������"
  "  "�  "-z�Ո����"��" *$$1""8�"�  "v����"
	 �����"
  "  "�  "-z����ף"��" *$$1""8�"�  "x�Ԣ��"
	 ������"
  "  "�  "-|������ϑ"��" *$$1""8�"�  "z�����"
	 ������"
  "  "�  "�����"
	 �ކ���"P*Lgradient_tape/sequential/densenet169/conv4_block29_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1152,2,2]"�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block29_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1152]"�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block29_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1152]"�  "-}�Ϊ����"��" *$$1""8�"�  "z���"
	 �ڍ���"
  "  "�  "z�ȅ�"
	 ������"
  "  "�  "v��"
	 �ֹ���"
  "  "�  "v�ם��"
	 ������"
  "  "�  "-zؠ����ą"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-z��Ǝ����"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  ".���ɒ����"��" *$$1""8�"�  "z�����"
	 ������"
  "  "�  "������"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block28_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~�������"��" *$$1""8�"�  "����"
	 ������"C*?gradient_tape/sequential/densenet169/conv4_block28_concat/Slice"
 �������"
*output" "*[200,1120,2,2]"�  "-~����"��" *$$1""8�"�  "z�����"
	 �ކ���"
  "  "�  "���ۡ�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block28_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "���̣�"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block28_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "���즦"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block28_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block28_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�ȥ���"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block28_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-RОǼ����"��" *$$1""8�"�  "���ۿ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S���Ǧ���"��" *$$1""8�"�  ".����ʦ��"��" *$$1""8�"�  ".����ͦб�"��" *$$1""8�"�  "�Л�Ц"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����Ц"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���Ѧ"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block28_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��Ѧ"
	 �줺��"
  "  "�  "-z�ο֦�׫"��" *$$1""8�"�  "x���٦"
	 ������"
  "  "�  "-|���ۦ���"��" *$$1""8�"�  "x�ȏަ"
	 ������"
  "  "�  "���ߦ"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv4_block28_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block28_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "��·�"
	 �����"P*Lgradient_tape/sequential/densenet169/conv4_block28_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}�������"��" *$$1""8�"�  "x����"
	 ������"
  "  "�  "x����"
	 ������"
  "  "�  "v����"
	 �����"
  "  "�  "v�Ŗ�"
	 �����"
  "  "�  "�����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block28_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1120,128]"�  "����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block28_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1120,1,1]"�  "��Υ�"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block28_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2272]"�  ".�������؋"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".������ȥ�"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  "-R��΁����"��" *$$1""8�"�  "���烧"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block28_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���ń�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block28_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���Ɔ�"
	 �ކ���"X*Tgradient_tape/sequential/densenet169/conv4_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1120,2,2]"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1120,1,1]"�  "-RȤȊ��"��" *$$1""8�"�  "���፧"
	 �ڍ���"X*Tgradient_tape/sequential/densenet169/conv4_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1120,2,2]"�  "������"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S��蕧���"��" *$$1""8�"�  ".�𵃙�੸"��" *$$1""8�"�  ".��㚛����"��" *$$1""8�"�  "���ӝ�"
	 �ކ���"X*Tgradient_tape/sequential/densenet169/conv4_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�荞��"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���О�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block28_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 ��Ҩ��"
  "  "�  "-z��������"��" *$$1""8�"�  "v��Ԧ�"
	 ������"
  "  "�  "-zହ����"��" *$$1""8�"�  "v�����"
	 �����"
  "  "�  "-zȜ���؊�"��" *$$1""8�"�  "x���"
	 ������"
  "  "�  "-|��������"��" *$$1""8�"�  "z�����"
	 �̝���"
  "  "�  "�؋���"
	 �ކ���"P*Lgradient_tape/sequential/densenet169/conv4_block28_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1120,2,2]"�  "���ո�"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block28_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1120]"�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block28_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1120]"�  "-}�䟼��Ť"��" *$$1""8�"�  "z��忧"
�2�?" ����" ���" ���"
	 �ڍ���"
  "  "�  "z�����"
	 ������"
  "  "�  "v�����"
	 ������"
  "  "�  "v諥§"
	 ������"
  "  "�  "-z���ŧ�˚"��" *$$1""8�"�  "v���ȧ"
	 ������"
  "  "�  "-z��˧���"��" *$$1""8�"�  "v���Χ"
	 ������"
  "  "�  ".���ϧ���"��" *$$1""8�"�  "z���ҧ"
	 ������"
  "  "�  "���ӧ"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block27_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~���ԧ؃�"��" *$$1""8�"�  "��ȝا"
	 ��ǚ��"C*?gradient_tape/sequential/densenet169/conv4_block27_concat/Slice"
 �������"
*output" "*[200,1088,2,2]"�  "-~���٧��"��" *$$1""8�"�  "z���ۧ"
	 �ކ���"
  "  "�  "��ݧ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block27_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "��ˌ�"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block27_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�ȧ��"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block27_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block27_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block27_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ל��"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-Rؠ���Ț�"��" *$$1""8�"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S��݄����"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".���Ɋ�Ƚ�"��" *$$1""8�"�  "�ة���"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���Ǎ�"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���Ŏ�"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block27_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 �줺��"
  "  "�  "-z��������"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "-|��ї����"��" *$$1""8�"�  "x�����"
	 �����"
  "  "�  "��́��"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv4_block27_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block27_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "�؁���"
	 �����"P*Lgradient_tape/sequential/densenet169/conv4_block27_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}�ġ��Т�"��" *$$1""8�"�  "x�ߙ��"
	 ������"
  "  "�  "x��̥�"
	 �����"
  "  "�  "v��륨"
	 ������"
  "  "�  "v�ݔ��"
	 ������"
  "  "�  "���է�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block27_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1088,128]"�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block27_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1088,1,1]"�  "������"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block27_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2272]"�  ".�������"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".���⹨���"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  "-R��������"��" *$$1""8�"�  "�Њ���"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block27_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block27_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����è"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1088,2,2]"�  "����Ũ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1088,1,1]"�  "-R��ƨ�ɓ"��" *$$1""8�"�  "����ɨ"
	 �ކ���"X*Tgradient_tape/sequential/densenet169/conv4_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1088,2,2]"�  "����̨"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S���Ш���"��" *$$1""8�"�  ".����Ө���"��" *$$1""8�"�  ".����ը���"��" *$$1""8�"�  "����ب"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���٨"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����ڨ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block27_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x���ۨ"
	 ��Ҩ��"
  "  "�  "-zȜ�ߨ���"��" *$$1""8�"�  "v����"
	 ������"
  "  "�  "-z�������"��" *$$1""8�"�  "v����"
	 �����"
  "  "�  "-z�������"��" *$$1""8�"�  "x�˭�"
	 ������"
  "  "�  "-|�����؞�"��" *$$1""8�"�  "z����"
E�?" ����" ���" ���"
	 ������"
  "  "�  "�����"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block27_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1088,2,2]"�  "�Ȕ��"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block27_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1088]"�  "�����"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block27_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1088]"�  "-}�ۙ�����"��" *$$1""8�"�  "z�ܒ��"
	 �ކ���"
  "  "�  "z�����"
	 ������"
  "  "�  "v�����"
	 ������"
  "  "�  "v�ԩ��"
	 ������"
  "  "�  "-z�����ʃ"��" *$$1""8�"�  "vЩ���"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "vЬ���"
	 ������"
  "  "�  ".���������"��" *$$1""8�"�  "z��"
	 ��ǚ��"
  "  "�  "���"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block26_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~��������"��" *$$1""8�"�  "�ȅҒ�"
	 ��ǚ��"C*?gradient_tape/sequential/densenet169/conv4_block26_concat/Slice"
 �������"
*output" "*[200,1056,2,2]"�  "-~��̓�ؚ�"��" *$$1""8�"�  "zؐ���"
	 ������"
  "  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block26_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "������"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block26_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "������"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block26_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block26_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���Ԯ�"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block26_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���Ͱ�"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R������"��" *$$1""8�"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S��ܽ����"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".���é���"��" *$$1""8�"�  "����ũ"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����Ʃ"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����Ʃ"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block26_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x���ǩ"
	 �줺��"
  "  "�  "-z���̩���"��" *$$1""8�"�  "x���Щ"
	 ������"
  "  "�  "-|���ѩ؃�"��" *$$1""8�"�  "x���ԩ"
	 ������"
  "  "�  "����թ"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv4_block26_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block26_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "����֩"
	 �����"P*Lgradient_tape/sequential/densenet169/conv4_block26_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}Ь�٩���"��" *$$1""8�"�  "x���ݩ"
	 ������"
  "  "�  "xЈ�ݩ"
	 �����"
  "  "�  "v���ީ"
	 ������"
  "  "�  "v��ߩ"
	 ������"
  "  "�  "�����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block26_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1056,128]"�  "�����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block26_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1056,1,1]"�  "�����"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block26_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2272]"�  ".�����Ѻ"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".�襂��ʵ"��" *$$1""8�"�  ".�������Ƕ"��" *$$1""8�"�  "-R�������"��" *$$1""8�"�  "��ܩ��"
	 ��؄��"Y*Ugradient_tape/sequential/densenet169/conv4_block26_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block26_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�誌��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1056,2,2]"�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1056,1,1]"�  "-R�����أ�"��" *$$1""8�"�  "������"
	 �ކ���"X*Tgradient_tape/sequential/densenet169/conv4_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1056,2,2]"�  "������"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S؆ъ����"��" *$$1""8�"�  ".�轣��࠻"��" *$$1""8�"�  ".��ڸ��Ⱥ�"��" *$$1""8�"�  "���풪"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���铪"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block26_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��Ɣ�"
	 ��Ҩ��"
  "  "�  "-zȆ������"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "v�����"
	 �����"
  "  "�  "-zȽ������"��" *$$1""8�"�  "x��"
	 ������"
  "  "�  "-|���蒠"��" *$$1""8�"�  "z�����"
	 ������"
  "  "�  "��Ꭼ�"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block26_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1056,2,2]"�  "�س٬�"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block26_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1056]"�  "��Ԃ��"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block26_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1056]"�  "-}�ˑ���֮"��" *$$1""8�"�  "z��޳�"
	 �ކ���"
  "  "�  "z�����"
	 ������"
  "  "�  "v�����"
	 ������"
  "  "�  "v�ψ��"
	 ������"
  "  "�  "-z�����蛝"��" *$$1""8�"�  "v��޼�"
	 ������"
  "  "�  "-z��׿����"��" *$$1""8�"�  "v���ª"
	 ������"
  "  "�  ".���êȟ�"��" *$$1""8�"�  "z��ƪ"
	 ��ǚ��"
  "  "�  "����Ȫ"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block25_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~໶ɪ��"��" *$$1""8�"�  "����̪"
	 ��ǚ��"C*?gradient_tape/sequential/densenet169/conv4_block25_concat/Slice"
 �������"
*output" "*[200,1024,2,2]"�  "-~І�ͪа�"��" *$$1""8�"�  "z�ֲЪ"
	 ������"
  "  "�  "�Ȟ�Ѫ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block25_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "����Ԫ"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block25_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "����ת"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block25_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block25_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block25_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ù�"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R؂�����"��" *$$1""8�"�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S�����ؠ�"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".�������޾"��" *$$1""8�"�  "�А���"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ��؄��"X*Tgradient_tape/sequential/densenet169/conv4_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block25_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�ȉ��"
	 �줺��"
  "  "�  "-z��八ȅ�"��" *$$1""8�"�  "x�눫"
	 ������"
  "  "�  "-|�򾊫���"��" *$$1""8�"�  "x�󓍫"
	 ������"
  "  "�  "�ȡ���"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv4_block25_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block25_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "������"
	 �����"P*Lgradient_tape/sequential/densenet169/conv4_block25_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}��ӓ�軼"��" *$$1""8�"�  "x𧽗�"
	 ������"
  "  "�  "x���"
	 ������"
  "  "�  "v�˜��"
	 ��˦��"
  "  "�  "v��ɘ�"
	 ������"
  "  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block25_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[1,1,1024,128]"�  "���ϛ�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block25_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[128,1024,1,1]"�  "��ֲ��"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block25_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2264]"�  ".���Ħ����"��" *$$1""8�"�  ".������"��" *$$1""8�"�  ".��ݨ�����"��" *$$1""8�"�  ".��ථ����"��" *$$1""8�"�  "-R�׵��غ�"��" *$$1""8�"�  "���˳�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block25_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block25_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��﫶�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*[200,1024,2,2]"�  "��鷫"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[128,1024,1,1]"�  "-R��޸����"��" *$$1""8�"�  "��𞼫"
	 �ކ���"X*Tgradient_tape/sequential/densenet169/conv4_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[200,1024,2,2]"�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S��ī�˚"��" *$$1""8�"�  ".�ȅ�ǫ���"��" *$$1""8�"�  ".����ɫ�˲"��" *$$1""8�"�  "����˫"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����̫"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����ͫ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block25_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��Ϋ"
	 ��Ҩ��"
  "  "�  "-z���ҫ���"��" *$$1""8�"�  "v���ի"
	 ������"
  "  "�  "-z���ث���"��" *$$1""8�"�  "v��۫"
	 �����"
  "  "�  "-z���ޫ���"��" *$$1""8�"�  "x�ߢ�"
H�`�?" ����" �� " �� "
	 ������"
  "  "�  "-|�����݂"��" *$$1""8�"�  "z����"
	 ������"
  "  "�  "�����"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block25_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*[200,1024,2,2]"�  "�����"
	 �Ʈ���"P*Lgradient_tape/sequential/densenet169/conv4_block25_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1024]"�  "�����"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block25_0_bn/FusedBatchNormGradV3"
 �������"
*output" "
*[1024]"�  "-}����Ю�"��" *$$1""8�"�  "z����"
	 �ކ���"
  "  "�  "z�ݧ�"
	 ������"
  "  "�  "v���"
	 ��ʦ��"
  "  "�  "vЃ��"
	 ������"
  "  "�  "-z�������"��" *$$1""8�"�  "v�ފ��"
	 �Ʈ���"
  "  "�  "-z��������"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  ".��Ѹ�����"��" *$$1""8�"�  "z�ꂁ�"
	 ��ǚ��"
  "  "�  "��Ѥ��"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block24_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~�ۄ����"��" *$$1""8�"�  "���̈�"
	 ��ǚ��"C*?gradient_tape/sequential/densenet169/conv4_block24_concat/Slice"
 �������"
*output" "*
	 ������"
  "  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block24_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "������"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block24_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "��ʵ��"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block24_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block24_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���桬"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block24_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R��觬���"��" *$$1""8�"�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S��沬���"��" *$$1""8�"�  ".���϶����"��" *$$1""8�"�  ".���︬���"��" *$$1""8�"�  "��ԩ��"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���ﻬ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��⟼�"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block24_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 �줺��"
  "  "�  "-z���¬���"��" *$$1""8�"�  "x�ޚŬ"
	 ������"
  "  "�  "-|��Ƭ��"��" *$$1""8�"�  "xȿ�ɬ"
	 ������"
  "  "�  "���ʬ"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv4_block24_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block24_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "����ˬ"
	 �����"P*Lgradient_tape/sequential/densenet169/conv4_block24_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}���ά���"��" *$$1""8�"�  "x���Ҭ"
	 ������"
  "  "�  "x���Ҭ"
	 ������"
  "  "�  "v���Ӭ"
	 �����"
  "  "�  "v���Ԭ"
	 �����"
  "  "�  "�П�֬"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block24_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block24_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block24_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2264]"�  ".��������"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".������"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  "-Rȱ��Ț�"��" *$$1""8�"�  "�����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block24_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block24_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �ކ���"X*Tgradient_tape/sequential/densenet169/conv4_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-SȆ�����"��" *$$1""8�"�  ".�ࢢ����"��" *$$1""8�"�  ".�ȣȆ����"��" *$$1""8�"�  "��̅��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�زԉ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�؟���"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block24_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x��䊭"
	 ��Ҩ��"
  "  "�  "-z�������"��" *$$1""8�"�  "v�ٷ��"
	 ������"
  "  "�  "-z�����ؿ�"��" *$$1""8�"�  "v��Ŗ�"
	 �����"
  "  "�  "-z��������"��" *$$1""8�"�  "x��Ǜ�"
	 ������"
  "  "�  "-|Чߞ����"��" *$$1""8�"�  "z��䡭"
	 ������"
  "  "�  "����"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block24_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block24_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[992]"�  "�ج���"
	 �Ʈ���"P*Lgradient_tape/sequential/densenet169/conv4_block24_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[992]"�  "-}��������"��" *$$1""8�"�  "z�����"
	 �ކ���"
  "  "�  "z��⫭"
	 �����"
  "  "�  "v�䑬�"
	 �Қ���"
  "  "�  "v�ᴬ�"
	 ������"
  "  "�  "-z��а����"��" *$$1""8�"�  "v��ճ�"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "v��渭"
	 �Ʈ���"
  "  "�  ".��餺����"��" *$$1""8�"�  "z��뼭"
	 ��ǚ��"
  "  "�  "��Ό��"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block23_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~ȅ�����"��" *$$1""8�"�  "����­"
	 ��ǚ��"C*?gradient_tape/sequential/densenet169/conv4_block23_concat/Slice"
 �������"
*output" "*
	 ������"
  "  "�  "�裶ǭ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block23_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "�ؙ�ʭ"
��a��?" ����" ��	" ��	"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block23_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "����ͭ"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block23_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
��a��?" ����" ���" ���"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block23_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����ݭ"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block23_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���߭"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-RІ���ߖ"��" *$$1""8�"�  "����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S�د����"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  "������"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block23_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�Ȅ��"
	 �줺��"
  "  "�  "-z�������"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "-|��������"��" *$$1""8�"�  "x��낮"
	 �����"
  "  "�  "����"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv4_block23_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block23_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "��ɠ��"
	 �����"P*Lgradient_tape/sequential/densenet169/conv4_block23_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}�߼���λ"��" *$$1""8�"�  "x�󟍮"
	 ������"
  "  "�  "x��ԍ�"
	 ������"
  "  "�  "v�����"
	 �����"
  "  "�  "v�����"
	 �����"
  "  "�  "���ɏ�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block23_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block23_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block23_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2264]"�  ".���������"��" *$$1""8�"�  ".���؝����"��" *$$1""8�"�  ".��򐠮���"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  "-R��է����"��" *$$1""8�"�  "����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block23_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���Ӫ�"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block23_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���Ӭ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �ކ���"X*Tgradient_tape/sequential/densenet169/conv4_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S��ɹ��ٓ"��" *$$1""8�"�  ".���ۼ���"��" *$$1""8�"�  ".����î���"��" *$$1""8�"�  "����Ʈ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�টǮ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "����Ȯ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block23_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�ےɮ"
	 ��Ҩ��"
  "  "�  "-zൈͮ���"��" *$$1""8�"�  "v�ԔЮ"
	 ������"
  "  "�  "-zЂ�Ӯ���"��" *$$1""8�"�  "v���֮"
	 �����"
  "  "�  "-z���ٮ���"��" *$$1""8�"�  "x���ܮ"
	 ������"
  "  "�  "-|���ݮ���"��" *$$1""8�"�  "z����"
	 ������"
  "  "�  "����"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block23_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block23_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[960]"�  "�����"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block23_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[960]"�  "-}�ީ���"��" *$$1""8�"�  "z����"
	 �ކ���"
  "  "�  "z����"
	 ������"
  "  "�  "v�����"
	 ������"
  "  "�  "v�����"
	 ������"
  "  "�  "-z�������"��" *$$1""8�"�  "v���"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  ".��ˉ�����"��" *$$1""8�"�  "z�����"
	 ��ǚ��"
  "  "�  "������"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block22_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~������"��" *$$1""8�"�  "������"
	 ��ǚ��"C*?gradient_tape/sequential/densenet169/conv4_block22_concat/Slice"
 �������"
*output" "*
	 ������"
  "  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block22_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "������"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block22_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "�୘��"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block22_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block22_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block22_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���ס�"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R��Ƥ����"��" *$$1""8�"�  "���ߧ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S��ɯ����"��" *$$1""8�"�  ".���貯���"��" *$$1""8�"�  ".�������ú"��" *$$1""8�"�  "���·�"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��司�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block22_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 �줺��"
  "  "�  "-zఄ�����"��" *$$1""8�"�  "x���"
	 ������"
  "  "�  "-|�Ӂï���"��" *$$1""8�"�  "x���Ư"
	 ������"
  "  "�  "����ǯ"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv4_block22_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block22_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "����ȯ"
	 �����"P*Lgradient_tape/sequential/densenet169/conv4_block22_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}Љ�˯�Ʈ"��" *$$1""8�"�  "x��ϯ"
	 ������"
  "  "�  "x���ϯ"
	 ������"
  "  "�  "v���ϯ"
	 ��Ǧ��"
  "  "�  "v���ѯ"
	 �����"
  "  "�  "����ү"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block22_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
���?" ����" ��" ��"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block22_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
���?" ����" �" �"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv4_block22_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2264]"�  ".����ݯ�ٺ"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".�د�����"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  "-R�������"��" *$$1""8�"�  "�����"
���?" ����" �" �"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv4_block22_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block22_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �ކ���"X*Tgradient_tape/sequential/densenet169/conv4_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S������ܗ"��" *$$1""8�"�  ".�������Լ"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  "���υ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���Ɇ�"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block22_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 ��Ҩ��"
  "  "�  "-z�֍���Ɣ"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-z�ΐ��ͩ"��" *$$1""8�"�  "v��ԓ�"
	 �����"
  "  "�  "-z�ᒖ����"��" *$$1""8�"�  "xВݘ�"
	 ������"
  "  "�  "-|��困�"��" *$$1""8�"�  "z����"
	 �̑���"
  "  "�  "�����"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block22_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block22_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[928]"�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block22_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[928]"�  "-}Г������"��" *$$1""8�"�  "z��ا�"
	 �ކ���"
  "  "�  "z�󗨰"
	 ������"
  "  "�  "v�����"
	 ������"
  "  "�  "v�訰"
	 ������"
  "  "�  "-z��ɭ����"��" *$$1""8�"�  "vȷϰ�"
	 ������"
  "  "�  "-z��������"��" *$$1""8�"�  "v��嵰"
	 ������"
  "  "�  ".���������"��" *$$1""8�"�  "z��蹰"
	 ��ǚ��"
  "  "�  "��ʋ��"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block21_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~�ӟ���ȉ"��" *$$1""8�"�  "���ؿ�"
	 ��ǚ��"C*?gradient_tape/sequential/densenet169/conv4_block21_concat/Slice"
 �������"
*output" "*
	 ������"
  "  "�  "����İ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block21_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "����ư"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block21_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "����ʰ"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block21_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block21_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����۰"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block21_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����ݰ"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R���߰���"��" *$$1""8�"�  "�����"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S�Į����"��" *$$1""8�"�  ".�������"��" *$$1""8�"�  ".�������"��" *$$1""8�"�  "������"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block21_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 �줺��"
  "  "�  "-z��������"��" *$$1""8�"�  "x�ܢ��"
	 ������"
  "  "�  "-|��������"��" *$$1""8�"�  "x��Ȁ�"
	 ������"
  "  "�  "���ׁ�"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv4_block21_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block21_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "���߃�"
	 �����"P*Lgradient_tape/sequential/densenet169/conv4_block21_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}Ф������"��" *$$1""8�"�  "xد���"
	 ������"
  "  "�  "x��勱"
	 ������"
  "  "�  "v���"
	 ��ȫ��"
  "  "�  "v�����"
	 ��ȫ��"
  "  "�  "���㍱"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block21_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block21_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 �����"Y*Ugradient_tape/sequential/densenet169/conv4_block21_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2256]"�  ".���똱���"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".���ɞ��۲"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  "-R��������"��" *$$1""8�"�  "�ȗ§�"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv4_block21_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block21_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ȡ��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �ކ���"X*Tgradient_tape/sequential/densenet169/conv4_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S��������"��" *$$1""8�"�  ".���������"��" *$$1""8�"�  ".��ŭ���ӿ"��" *$$1""8�"�  "���꾱"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�莱��"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ކ��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block21_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x�����"
	 ��Ҩ��"
  "  "�  "-z��ű���"��" *$$1""8�"�  "v�Ջɱ"
	 ������"
  "  "�  "-z���˱��"��" *$$1""8�"�  "v���α"
	 �����"
  "  "�  "-z���ѱ���"��" *$$1""8�"�  "x��ӱ"
	 ������"
  "  "�  "-|��ձ�"��" *$$1""8�"�  "z��ױ"
	 ������"
  "  "�  "����ر"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block21_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block21_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[896]"�  "����۱"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block21_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[896]"�  "-}໵ޱ���"��" *$$1""8�"�  "z�ԙ�"
	 �ކ���"
  "  "�  "zȡ��"
	 ������"
  "  "�  "v����"
	 ��Ʀ��"
  "  "�  "v����"
���?" ����" �" �"
	 ������"
  "  "�  "-z�����ʃ"��" *$$1""8�"�  "v����"
���?" ����" �" �"
	 ������"
  "  "�  "-z�������"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  ".�������"��" *$$1""8�"�  "z���"
	 ��ǚ��"
  "  "�  "������"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block20_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~������"��" *$$1""8�"�  "��ʽ��"
	 ��ǚ��"C*?gradient_tape/sequential/densenet169/conv4_block20_concat/Slice"
 �������"
*output" "*
	 ������"
  "  "�  "������"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block20_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "������"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block20_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "��Ɠ��"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block20_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block20_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "������"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block20_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "���"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R؇����ڢ"��" *$$1""8�"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S������֙"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  ".�������۷"��" *$$1""8�"�  "���ګ�"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ᡬ�"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "���Ѭ�"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block20_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "xȟˮ�"
	 �줺��"
  "  "�  "-z������ܬ"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "-|��������"��" *$$1""8�"�  "x�̥��"
	 ������"
  "  "�  "������"
	 ��Ҩ��"P*Lgradient_tape/sequential/densenet169/conv4_block20_1_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block20_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "��ܱ��"
	 �����"P*Lgradient_tape/sequential/densenet169/conv4_block20_1_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[128]"�  "-}��ǿ��ǳ"��" *$$1""8�"�  "xر�ò"
	 ������"
  "  "�  "x���Ĳ"
	 ������"
  "  "�  "v��Ų"
	 ��ȫ��"
  "  "�  "vȉ�Ų"
	 ��ȫ��"
  "  "�  "��Áǲ"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block20_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block20_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*
	 �����"Y*Ugradient_tape/sequential/densenet169/conv4_block20_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
*[2256]"�  ".��ǲҲ���"��" *$$1""8�"�  ".����ղ���"��" *$$1""8�"�  ".�୙ز���"��" *$$1""8�"�  ".����ڲ���"��" *$$1""8�"�  "-R�ݲ���"��" *$$1""8�"�  "����߲"
	 �����"Y*Ugradient_tape/sequential/densenet169/conv4_block20_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�Я��"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block20_1_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "�ȁ��"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �ކ���"X*Tgradient_tape/sequential/densenet169/conv4_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S�����"��" *$$1""8�"�  ".������Ȩ�"��" *$$1""8�"�  ".��������"��" *$$1""8�"�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ȧ��"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "������"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block20_1_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "x輵��"
	 ��Ҩ��"
  "  "�  "-z�������"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-z�Ն�؏�"��" *$$1""8�"�  "v�Ǳ��"
	 �����"
  "  "�  "-z��������"��" *$$1""8�"�  "x�����"
	 ������"
  "  "�  "-|�ċ�����"��" *$$1""8�"�  "z��䒳"
	 ������"
  "  "�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block20_0_bn/FusedBatchNormGradV3"
 �������"
*output" "*
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block20_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[864]"�  "������"
	 ������"P*Lgradient_tape/sequential/densenet169/conv4_block20_0_bn/FusedBatchNormGradV3"
 �������"
*output" "	*[864]"�  "-}��������"��" *$$1""8�"�  "z�Ѷ��"
	 �ކ���"
  "  "�  "z��ǝ�"
	 ������"
  "  "�  "v��읳"
	 �꓉��"
  "  "�  "vإ���"
	 �����"
  "  "�  "-z��������"��" *$$1""8�"�  "v�����"
	 ������"
  "  "�  "-z�׋��Ф�"��" *$$1""8�"�  "vА㩳"
	 ������"
  "  "�  ".�螬�����"��" *$$1""8�"�  "z�����"
	 ��ǚ��"
  "  "�  "������"
	 �줺��"E*Agradient_tape/sequential/densenet169/conv4_block19_concat/Slice_1"
 �������"
*output" "*[200,32,2,2]"�  "-~��������"��" *$$1""8�"�  "���ɵ�"
	 ��ǚ��"C*?gradient_tape/sequential/densenet169/conv4_block19_concat/Slice"
 �������"
*output" "*
	 ������"
  "  "�  "���껳"
	 ������"Y*Ugradient_tape/sequential/densenet169/conv4_block19_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"
*output" "*[3,3,128,32]"�  "���׽�"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block19_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "*[32,128,3,3]"�  "������"
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block19_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"*temp" "
	 ��í��"Y*Ugradient_tape/sequential/densenet169/conv4_block19_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "��ٜг"
	 ��Ҩ��"Y*Ugradient_tape/sequential/densenet169/conv4_block19_2_conv/Conv2D/Conv2DBackpropFilter"
 �������"  "�  "����ҳ"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block19_2_conv/Conv2D/Conv2DBackpropInput"
 �������"
*output" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block19_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*[32,128,3,3]"�  "-R���ֳ���"��" *$$1""8�"�  "����ٳ"
	 ������"X*Tgradient_tape/sequential/densenet169/conv4_block19_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "*
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block19_2_conv/Conv2D/Conv2DBackpropInput"
 �������"*temp" "
*[2208]"�  "-S�������"��" *$$1""8�"�  ".����࠻"��" *$$1""8�"�  ".�ȓ�����"��" *$$1""8�"�  "����"
	 ��Ҩ��"X*Tgradient_tape/sequential/densenet169/conv4_block19_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "�����"
	 �����"X*Tgradient_tape/sequential/densenet169/conv4_block19_2_conv/Conv2D/Conv2DBackpropInput"
 �������"  "�  "��ݓ�"
 �������"  "�  "x���"
	 �줺��"
	 ������"