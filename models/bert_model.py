from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn.functional as F
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50
import sys
sys.path.append('/home/nlp/code/HVPNeT/models')
from model.attention.SEAttention import SEAttention
from model.attention.SKAttention import SKAttention
from model.attention.CBAM import CBAMBlock
from model.attention.BAM import BAMBlock
from model.attention.ECAAttention import ECAAttention
from model.attention.DANet import DAModule
from model.attention.PSA import PSA
from model.attention.ShuffleAttention import ShuffleAttention
from model.attention.EMSA import EMSA
from model.attention.MUSEAttention import MUSEAttention
from model.attention.SGE import SpatialGroupEnhance
from model.attention.A2Atttention import DoubleAttention
from model.attention.AFT import AFT_FULL
from model.attention.OutlookAttention import OutlookAttention
from model.attention.CoTAttention import CoTAttention
from model.attention.ResidualAttention import ResidualAttention
from model.attention.S2Attention import S2Attention
from model.attention.TripletAttention import TripletAttention
from model.attention.ParNetAttention import *
# from model.attention.ACmix import ACmix
from model.attention.Axial_attention import AxialImageTransformer
import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
from torchcrf import CRF

import torch
from torch import nn
from .file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'
def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model

logger = logging.getLogger(__name__)
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # Backward compatibility with old naming format
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        # self.resnet = resnet152(pretrained=True)
        # net = getattr(resnet, 'resnet152')()
        # net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152-b121ed2d.pth')))
        # self.resnet = myResnet(net, args.fine_tune_cnn, args.device)
        # self.resnet.to(args.device)
        # 这一行创建了一个名为 resnet 的属性，它将存储一个 ResNet-50 模型的实例。pretrained=True
        # 意味着加载已经在大规模图像数据上预训练过的模型权重。这个模型将在后续的前向传播中使用，用于图像数据的特征提取。
    def forward(self, x, aux_imgs=None):
        # full image prompt
        prompt_guids = self.get_resnet_prompt(x)# 这一行代码调用了 self.get_resnet_prompt(x) 方法，将输入 x 送入 ResNet 模型，
                                                # 获取图像特征。prompt_guids 存储了这些特征，维度是 [bsz, 4, 256, 7, 7]，
                                                # 其中 bsz 是批处理大小，4 表示使用 ResNet 提取的四个不同的层次的特征
        if aux_imgs is not None: #这一行代码检查是否提供了辅助图像数据 aux_imgs。aux_imgs 是一个包含额外图像数据的张量。
            aux_prompt_guids = []   #这一行创建一个空列表 aux_prompt_guids，用于存储辅助图像数据的特征。
                                    # goal: 3 x (4 x [bsz, 256, 7, 7])
            aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])  #这一行对 aux_imgs 进行维度变换，将原来的形状 [bsz, 3, 3, 224, 224]
                                                          # 转换为 [3, bsz, 3, 224, 224]。这通常用于调整数据的维度顺序，
                                                          # 以适应模型的输入要求。
            for i in range(len(aux_imgs)):                # 遍历 aux_imgs 中的每个辅助图像。
                aux_prompt_guid = self.get_resnet_prompt(aux_imgs[i]) # 4 x [bsz, 256, 7, 7]
                # 在循环中，对每个辅助图像 aux_imgs[i] 调用了 self.get_resnet_prompt 方法，以获取图像特征。
                # aux_prompt_guid 存储了这些特征，维度为 [4, bsz, 256, 7, 7]，其中 4 表示使用 ResNet 提取的四个不同的层次的特征。

                aux_prompt_guids.append(aux_prompt_guid)  #这一行将每个辅助图像的特征 aux_prompt_guid 添加到
                                                     # aux_prompt_guids 列表中。
            # for i in aux_prompt_guids:
            #      print()
            #      x1, x2, x3, x4 = self.cross_scale_attention(i[0], i[1], i[2], i[3])
            #      print(x1, x2, x3, x4)
            return prompt_guids, aux_prompt_guids
        # 如果存在辅助图像数据，此处返回两个值：主要图像的特征prompt_guids和辅助图像的特征列表aux_prompt_guids。这些特征将用于后续的多模态任务。
        return prompt_guids, None
    # 如果没有提供辅助图像数据，仅返回主要图像的特征prompt_guids，而aux_prompt_guids设置为None。
    def get_resnet_prompt(self, x):
        """generate image prompt

        Args:
            x ([torch.tenspr]): bsz x 3 x 224 x 224

        Returns:
            prompt_guids ([List[torch.tensor]]): 4 x List[bsz x 256 x 7 x 7]
        """
        # image: bsz x 3 x 224 x 224
        prompt_guids = [] #这一行创建一个空列表 prompt_guids，用于存储图像的特征。
        for name, layer in self.resnet.named_children(): #这一行通过遍历 ResNet 模型的各个子层次来迭代处理图像数据。
            if name == 'fc' or name == 'avgpool':  continue #这一行用于跳过ResNet模型中的全连接层和平均池化层。
                                                            # 这是因为在通常情况下，这些层次不用于提取特征。
            x = layer(x)    # 这一行将输入张量x通过当前层layer进行前向传播，从而提取特征。在ResNet中，
                            # 这些特征通常具有 256 个通道，维度为 [bsz, 256, 56, 56]。
            if 'layer' in name: #这一行检查当前层的名称是否包含 'layer'，以确定当前层是否为 ResNet 中的卷积层。
                                # 通常，只有卷积层的特征会用于生成图像的特征表示。
                bsz, channel, ft, _ = x.size() #这一行获取当前特征张量 x 的维度信息。bsz 表示批处理大小，channel 表示通道数，
                                               # ft 表示特征图的宽度（它会随着不同层次而变化）。
                kernel = ft // 2               #这一行计算用于平均池化的核大小 kernel。它的值是特征图宽度的一半，目的是进行下采样，
                                               # 将特征图尺寸减小。
                prompt_kv = nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x) # 这一行对特征张量 x 进行平均池化操作，
                                # 以减小特征图的尺寸。prompt_kv 存储了这些池化后的特征，维度是 [bsz, 256, 7, 7]。
                prompt_guids.append(prompt_kv)   # 这一行将池化后的特征 prompt_kv 添加到 prompt_guids 列表中。
                                # 这是在不同卷积层中提取的特征。
        return prompt_guids     #最后，函数返回存储了不同卷积层提取的特征的 prompt_guids 列表。
    def cross_scale_attention(self,x3, x4, x5, x6):
        h3, h4, h5, h6 = x3.shape[2], x4.shape[2], x5.shape[2], x6.shape[2]
        h_max = max(h3, h4, h5, h6)
        x3 = F.interpolate(x3, size=(h_max, h_max), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(h_max, h_max), mode='bilinear', align_corners=True)
        x5 = F.interpolate(x5, size=(h_max, h_max), mode='bilinear', align_corners=True)
        x6 = F.interpolate(x6, size=(h_max, h_max), mode='bilinear', align_corners=True)
        mul = x3 * x4 * x5 * x6
        x3 = x3 + mul
        x4 = x4 + mul
        x5 = x5 + mul
        x6 = x6 + mul
        x3 = F.interpolate(x3, size=(h3, h3), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(h4, h4), mode='bilinear', align_corners=True)
        x5 = F.interpolate(x5, size=(h5, h5), mode='bilinear', align_corners=True)
        x6 = F.interpolate(x6, size=(h6, h6), mode='bilinear', align_corners=True)
        return x3, x4, x5, x6

class HMNeTREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(HMNeTREModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(768*2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer

        if self.args.use_prompt:
            self.image_model = ImageModel()

            self.encoder_conv =  nn.Sequential(
                                    nn.Linear(in_features=3840, out_features=800),
                                    nn.Tanh(),
                                    nn.Linear(in_features=800, out_features=4*2*768)
                                )

            self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        images=None,
        aux_imgs=None,
    ):

        bsz = input_ids.size(0)
        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_guids = None
            prompt_attention_mask = attention_mask

        output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=prompt_attention_mask,
                    past_key_values=prompt_guids,
                    output_attentions=True,
                    return_dict=True
        )

        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits
        return logits

    def get_visual_prompt(self, images, aux_imgs):
        bsz = images.size(0)
        # full image prompt
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....
        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840

        # aux image prompts # 3 x (4 x [bsz, 256, 2, 2])
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]

        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 x [bsz, 4, 4*2*768]
        split_prompt_guids = prompt_guids.split(768*2, dim=-1)   # 4 x [bsz, 4, 768*2]
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4, 768*2]]

        sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2

        result = []
        for idx in range(12):  # 12
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            # use gate mix aux image prompts
            aux_key_vals = []   # 3 x [bsz, 4, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result


# class HMNeTNERModel(BertPreTrainedModel):
class HMNeTNERModel(nn.Module):
    def __init__(self, config, label_list, args, layer_num1=1): #这一行定义了 HMNeTNERModel 类的初始化方法，该方法接受两个参数：label_list 和 args。
        # 2023-10-25新增configlayer_num1=1,
        super(HMNeTNERModel, self).__init__() #这一行调用了父类 nn.Module 的初始化方法，确保正确初始化模型。
        self.args = args #这一行将传入的 args 参数赋值给模型的 self.args 成员变量，以便后续使用。
        self.prompt_dim = args.prompt_dim #这一行从参数 args 中提取了 prompt_dim，这个变量用来表示提示信息的维度。
        self.prompt_len = args.prompt_len #这一行从参数 args 中提取了 prompt_len，表示提示信息的长度。
        self.bert = BertModel.from_pretrained(args.bert_name) #这一行创建了一个 BERT 模型，并加载了预训练的权重。
        self.bert_config = self.bert.config #这一行提取了 BERT 模型的配置信息，包括模型的架构、隐藏层大小等。

        if args.use_prompt: #这一行检查是否启用了提示信息。如果 args.use_prompt 为真，则会执行以下代码块；否则，跳过。
            self.image_model = ImageModel()  #这一行创建了一个图像模型 ImageModel 的实例，用于处理图像数据。
            # 这个模型的输出是大小为 [bsz, 6, 56, 56] 的特征图。
            self.encoder_conv =  nn.Sequential(
                            nn.Linear(in_features=3840, out_features=800),
                            nn.Tanh(),
                            nn.Linear(in_features=800, out_features=4*2*768)
                            ) #这一行定义了一个由线性层和激活函数组成的编码器 encoder_conv。
            # 它接受输入维度为 3840（这个值可能是根据具体需求选择的）的特征并输出大小为 [bsz, 4*2*768] 的特征。
            self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])
        #这一行创建了一个包含12个线性层的模块列表gates，用于实现门控机制。每个线性层将输入维度为4*768*2的特征映射到维度为4的输出。
        # 这些线性层将用于控制多模态特征之间的信息流动。
        self.num_labels  = len(label_list)  #这一行计算了命名实体识别任务中的标签数量，并将结果存储在self.num_labels中。
        # 通常，这个值对应于不同的命名实体类别数量。
        # print(self.num_labels) #
        self.crf = CRF(self.num_labels, batch_first=True) #这一行创建了一个条件随机场（CRF）层，用于处理序列标注任务，
        # 如命名实体识别。self.num_labels 表示标签数量，batch_first=True 表示输入数据的第一个维度是批处理大小。

        self.fc = nn.Linear(768, self.num_labels) #这一行创建了一个线性层 fc，
        # 用于将 BERT 模型的输出特征映射到标签数量的维度上，以进行分类任务。
        self.dropout = nn.Dropout(0.1) #这一行创建了一个丢弃层 dropout，用于在模型训练过程中随机丢弃一部分神经元，以防止过拟合。
        self.vismap2text = nn.Linear(2048, 768)
        # 创建了一个线性层，用于将视觉嵌入（2048维）映射到文本隐藏表示的维度（BERT模型的隐藏维度）。
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        # # 创建了一个BertCrossEncoder实例，用于执行文本到图像的交叉注意力操作。
        self.crs_classifier = nn.Linear(768 * 2 * 128, 2)
        # 创建了一个线性层，用于分类任务的输出。这里的输入维度是config.hidden_size * 2 * 128，根据模型的不同设置，可能需要调整。
        self.soft = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        # 创建了Softmax和Sigmoid激活函数的实例，用于模型的激活操作。
        self.Gate_text = nn.Linear(768, 768)
        self.Gate_image = nn.Linear(768, 768)
        # 创建了两个线性层，用于模型中的门控操作。
        self.classifier = nn.Linear(768 * 2, 13)
        # 创建了一个线性层，用于标记分类任务的输出。输入维度是config.hidden_size * 2，根据模型的不同设置，可能需要调整。
        self.crs_loss = nn.CrossEntropyLoss()
        # 创建了一个交叉熵损失函数的实例，用于条件随机场（CRF）损失的计算。
        self.text_ouput_cl = nn.Linear(768, 768)
        self.image_dense_cl = nn.Linear(2048, 768)
        self.image_output_cl = nn.Linear(768, 768)
        # 创建了一系列线性层和激活函数，用于模型中的文本和图像特征的处理。
        self.text_dense_cl = nn.Linear(768, 768)
        self.relu = nn.ReLU()
        self.bert1 = BertModel1.from_pretrained(config)
    def text_toimage_loss(self,text_h1, image_h1, temp):
        # 这是一个类方法的定义，它接受三个参数：self（表示类的实例）、
        # text_h1（文本的隐藏表示）、image_h1（图像的隐藏表示）和temp（温度参数，用于控制损失的范围）。
        batch_size = text_h1.shape[0]
        # 计算text_h1中的样本数，这将是批处理的大小。
        loss = 0
        # 初始化损失值为0。
        for i in range(batch_size):
            # 循环遍历每个样本。
            up = torch.exp(
                (torch.matmul(text_h1[i], image_h1[i]) / (torch.norm(text_h1[i]) * torch.norm(image_h1[i]))) / temp
            )
            # 计算分子部分up，这是一个数学表达式，计算了两个表示之间的相似性，然后应用了指数函数。
            down = torch.sum(
                torch.exp((torch.sum(text_h1[i] * image_h1, dim=-1) / (
                            torch.norm(text_h1[i]) * torch.norm(image_h1, dim=1))) / temp), dim=-1)
            # 计算分母部分down，这也是一个数学表达式，计算了多个表示之间的相似性，然后应用了指数函数，并对它们进行求和。
            loss += -torch.log(up / down)
        # 计算并累加损失，这里使用了对数变换。
        return loss

        # 这个方法的作用是计算文本到图像的损失，其中损失值表示了文本和图像之间的相似性。
    def image_totext_loss(self,text_h1, image_h1, temp):
        # 这是一个类方法的定义，它接受三个参数：self（表示类的实例）、
        # text_h1（文本的隐藏表示）、image_h1（图像的隐藏表示）和temp（温度参数，用于控制损失的范围）。
        batch_size = text_h1.shape[0]
        # 计算text_h1中的样本数，这将是批处理的大小。
        loss = 0
        # 初始化损失值为0。
        for i in range(batch_size):
            # 循环遍历每个样本。
            up = torch.exp(
                (
                        torch.matmul(image_h1[i], text_h1[i]) / (torch.norm(image_h1[i]) * torch.norm(text_h1[i]))
                ) / temp
            )
            # 计算分子部分up，这是一个数学表达式，计算了两个表示之间的相似性，然后应用了指数函数。
            down = torch.sum(
                torch.exp((torch.sum(image_h1[i] * text_h1, dim=-1) / (
                            torch.norm(image_h1[i]) * torch.norm(text_h1, dim=1))) / temp), dim=-1)
            # 计算分母部分down，这也是一个数学表达式，计算了多个表示之间的相似性，然后应用了指数函数，并对它们进行求和。
            loss += -torch.log(up / down)
            # 计算并累加损失，这里使用了对数变换。
        return loss
        # 返回计算得到的损失值。这个方法的作用是计算图像到文本的损失，其中损失值表示了图像和文本之间的相似性。

    def total_loss(self,text_h1, image_h1, temp, temp_lamb):
        # 这是一个类方法的定义，接受四个参数：self（表示类的实例）、text_h1（文本的隐藏表示）、
        # image_h1（图像的隐藏表示）、temp（温度参数，用于控制损失的范围）和temp_lamb（用于控制文本和图像损失的权重）。
        lamb = temp_lamb
        # 将temp_lamb赋值给局部变量lamb，以便在后续计算中使用。
        batch_size = text_h1.shape[0]
        # 计算text_h1中的样本数，这将是批处理的大小。
        loss = (1 / batch_size) * (
                    lamb * self.text_toimage_loss(text_h1, image_h1, temp) + (1 - lamb) * self.image_totext_loss(text_h1, image_h1, temp))
        # 计算总的损失值，它由两部分组成：文本到图像的损失和图像到文本的损失，这两部分损失的权重由lamb和1 - lamb控制。损失值被除以批处理大小以获得平均损失。
        return loss
        # 返回计算得到的总损失值。这个方法的作用是根据文本到图像和图像到文本的损失以及权重，计算整体的损失值，用于模型的训练。
    # 2023-10-25新增损失函数
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None, visual_embeds_mean=None, visual_embeds_att=None,temp=None,
                temp_lamb=None,lamb=None, negative_rate=None):
        #2023-10-25新增visual_embeds_mean, visual_embeds_att,temp=None,temp_lamb=None,lamb=None, negative_rate=None
        aa, sequence_output_pooler = self.bert1(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                       output_all_encoded_layers=False)
        # 2023-11-2添加dropout
        sequence_output_pooler = self.dropout(sequence_output_pooler)
        # 2023-11-2添加dropout
        if self.args.use_prompt:  #这一行检查是否启用了提示信息。如果 self.args.use_prompt 为真，则会执行以下代码块；否则，跳过。
            prompt_guids = self.get_visual_prompt(images, aux_imgs)
            #这一行调用self.get_visual_prompt方法，用于生成视觉提示信息。视觉提示信息是来自图像的一种形式的信息，
            # 用于辅助文本和图像的信息提取任务。
            prompt_guids_length = prompt_guids[0][0].shape[2]  #这一行计算视觉提示信息的长度，通常是它的第一个维度的长度。
            # attention_mask: bsz, seq_len
            # prompt attention， attention mask
            bsz = attention_mask.size(0)  #这一行获取输入数据的批处理大小 bsz。
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            #这一行创建一个形状为 (bsz, prompt_guids_length) 的全 1 的张量，用于表示视觉提示信息的掩码。
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
            #这一行将视觉提示信息的掩码与文本输入的注意力掩码连接在一起，以获得一个综合的注意力掩码 prompt_attention_mask。
        else:  #如果未启用提示信息，执行以下代码块。
            prompt_attention_mask = attention_mask  #这一行将 prompt_attention_mask 设置为与文本输入的注意力掩码相同。
            prompt_guids = None  #如果未启用提示信息，将 prompt_guids 设置为 None。
        added_attention_mask = attention_mask.clone()
        bert_output  = self.bert(input_ids=input_ids,
                            attention_mask=prompt_attention_mask,
                            token_type_ids=token_type_ids,
                            past_key_values=prompt_guids,
                            return_dict=True,
                           )
        # aa, sequence_output_pooler = bert_output[:2]
        # print(aa,sequence_output_pooler)
        #这一行调用 BERT 模型的前向传播，传递文本输入 (input_ids、attention_mask、token_type_ids) 和生成的视觉提示信息
        # （如果已启用）。bert_output 包含了 BERT 模型的输出。
        sequence_output = bert_output['last_hidden_state']  # bsz, len, hidden
        #这一行从 BERT 输出中提取了最后一层的隐藏状态，表示为 sequence_output，它包含了文本序列的编码表示。
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        #这一行对文本编码进行丢弃（dropout）操作，以减轻过拟合的风险。
        # 2023-10-25新增
        # vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)
        # converted_vis_embed_map = self.vismap2text(vis_embed_map)
        # img_mask = added_attention_mask[:, :49]
        # extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        # extended_img_mask = extended_img_mask.to(dtype=torch.long)
        # extended_img_mask = (1.0 - extended_img_mask) * -10000.0
        # cross_encoder = self.txt2img_attention(sequence_output, converted_vis_embed_map, extended_img_mask)
        # cross_output_layer = cross_encoder[-1]
        # cross_output_layer_crs = cross_output_layer.clone()
        # labels_crs = torch.ones(sequence_output.shape[0], dtype=torch.long).cuda()
        # if (negative_rate != None and cross_output_layer.shape[0] > negative_rate):
        #     all_negative_samples = cross_output_layer_crs[(cross_output_layer.shape[0] - negative_rate):]
        #     front_negative_samples = all_negative_samples[:int(all_negative_samples.shape[0] / 2)]
        #     after_negative_samples = all_negative_samples[int(all_negative_samples.shape[0] / 2):]
        #     for i, n in enumerate(front_negative_samples):
        #         temp_samples = front_negative_samples[i].clone()
        #         front_negative_samples[i] = after_negative_samples[i].clone()
        #         after_negative_samples[i] = temp_samples.clone()
        #         # 这个循环将front_negative_samples  和after_negative_samples中的元素两两交换，以增加训练的多样性
        #     labels_crs[(cross_output_layer.shape[0] - negative_rate):] = 0
        # cross_output = cross_output_layer
        # if labels is not None:
        #     cross_output = cross_output_layer_crs
        # crs_result = self.crs_classifier(
        #     torch.cat((sequence_output, cross_output), dim=-1).view(sequence_output.shape[0], -1))
        # P = self.soft(crs_result)  # batch_size * 2
        # P = P[:, -1]
        # P = P.unsqueeze(-1).unsqueeze(-1)
        # new_cross_output_layer = P * cross_output
        # Gate = self.sigmoid((self.Gate_text(sequence_output) + self.Gate_image(new_cross_output_layer)))
        # gated_converted_att_vis_embed = Gate * new_cross_output_layer
        # final_output = torch.cat((sequence_output, gated_converted_att_vis_embed),
        #                          dim=-1)
        # bert_feats = self.classifier(final_output)
        # final_bert_feats = bert_feats.clone()
        # 2023-10-25新增
        emissions = self.fc(sequence_output)    # bsz, len, labels
        logits = self.crf.decode(emissions, attention_mask.byte())
        # 这一行使用条件随机场（CRF）解码 emissions，生成标签预测 logits，attention_mask.byte() 用于指示标签的有效位置。
        loss = None  # 初始化损失值 loss，默认为 None。

        #这一行将文本编码通过一个线性层 self.fc，将其映射到标签数量的维度，生成 emissions，这是用于条件随机场（CRF）的输入。
        if labels is not None:  #这一行检查是否提供了真实标签 labels，如果提供了，则执行以下代码块。
            # loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            crs_loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            # 这一行计算CRF模型的负对数似然损失。它将预测标签emissions与真实标签labels进行比较，使用attention_mask.byte()
            # 指示有效标签的位置，并将损失值保存在 loss 变量中。
            # crs_loss = self.crs_loss(crs_result.view(-1, 2), labels_crs.view(-1))
            # 2023-10-25新增

            text_output_cl = self.text_ouput_cl(self.relu(self.text_dense_cl(sequence_output_pooler)))
            image_ouput_cl = self.image_output_cl(self.relu(self.image_dense_cl(visual_embeds_mean)))
            cl_loss = self.total_loss(text_output_cl, image_ouput_cl, temp, temp_lamb)
            main_loss = - self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            # main_loss = - self.crf(final_bert_feats, labels, mask=attention_mask.byte(), reduction='mean')
            alpha = lamb
            aux_loss = crs_loss + cl_loss
            loss = alpha * main_loss + (1 - alpha) * aux_loss

            # 2023-10-25新增
            # loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            #这一行计算CRF模型的负对数似然损失。它将预测标签emissions与真实标签labels进行比较，使用attention_mask.byte()
            # 指示有效标签的位置，并将损失值保存在 loss 变量中。

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )



    #2023-10-25这一行返回模型的输出，其中包括损失值 loss 和标签预测 logits。

    def get_visual_prompt(self, images, aux_imgs):  #
        bsz = images.size(0)  #这一行获取输入图像 images 的批处理大小并存储在变量 bsz 中。
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....
        #这一行调用另一个模型self.image_model来从图像中提取提示信息。prompt_guids包含主要的提示信息，
        # 而aux_prompt_guids包含辅助提示信息。这些信息的形状通常为 [bsz, C, H, W]，其中 C 表示通道数，H 和 W 表示高度和宽度。
        # device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
        # net = getattr(resnet, 'resnet152')()
        # encoder = myResnet(net, False, device)
        # imgs_f, img_mean, img_att = encoder(images)
        # print(imgs_f, img_mean, img_att)
        # 2023-10-31修改
        # expanded_tensor = prompt_guids[0].expand([8,1024,2,2])
        # 2023-11-2修改SEAttention
        # prompt_guids[0] = F.interpolate(prompt_guids[0], size=(7, 7), mode='bilinear', align_corners=True)
        # se256 = SEAttention(channel=256, reduction=8)
        # se256.to(self.args.device)
        # prompt_guids[0] = se256(prompt_guids[0])
        # prompt_guids[0] = F.interpolate(prompt_guids[0], size=(2, 2), mode='bilinear', align_corners=True)
        # prompt_guids[1] = F.interpolate(prompt_guids[1], size=(7, 7), mode='bilinear', align_corners=True)
        # se512 = SEAttention(channel=512, reduction=8)
        # se512.to(self.args.device)
        # prompt_guids[1] = se512(prompt_guids[1])
        # prompt_guids[1] = F.interpolate(prompt_guids[1], size=(2, 2), mode='bilinear', align_corners=True)
        # prompt_guids[2] = F.interpolate(prompt_guids[2], size=(7, 7), mode='bilinear', align_corners=True)
        # se1024 = SEAttention(channel=1024, reduction=8)
        # se1024.to(self.args.device)
        # prompt_guids[2] = se1024(prompt_guids[2])
        # prompt_guids[2] = F.interpolate(prompt_guids[2], size=(2, 2), mode='bilinear', align_corners=True)
        # prompt_guids[3] = F.interpolate(prompt_guids[3], size=(7, 7), mode='bilinear', align_corners=True)
        # se2048 = SEAttention(channel=2048, reduction=8)
        # se2048.to(self.args.device)
        # prompt_guids[3] = se2048(prompt_guids[3])
        # prompt_guids[3] = F.interpolate(prompt_guids[3], size=(2, 2), mode='bilinear', align_corners=True)
        # 2023-11-2修改
        # 2023-11-2修改SKAttention
        prompt_guids[0] = F.interpolate(prompt_guids[0], size=(7, 7), mode='bilinear', align_corners=True)
        sk256 = SKAttention(channel=256, reduction=8)
        sk256.to(self.args.device)
        prompt_guids[0] = sk256(prompt_guids[0])
        prompt_guids[0] = F.interpolate(prompt_guids[0], size=(2, 2), mode='bilinear', align_corners=True)
        prompt_guids[1] = F.interpolate(prompt_guids[1], size=(7, 7), mode='bilinear', align_corners=True)
        sk512 = SKAttention(channel=512, reduction=8)
        sk512.to(self.args.device)
        prompt_guids[1] = sk512(prompt_guids[1])
        prompt_guids[1] = F.interpolate(prompt_guids[1], size=(2, 2), mode='bilinear', align_corners=True)
        prompt_guids[2] = F.interpolate(prompt_guids[2], size=(7, 7), mode='bilinear', align_corners=True)
        sk1024 = SKAttention(channel=1024, reduction=8)
        sk1024.to(self.args.device)
        prompt_guids[2] = sk1024(prompt_guids[2])
        prompt_guids[2] = F.interpolate(prompt_guids[2], size=(2, 2), mode='bilinear', align_corners=True)
        prompt_guids[3] = F.interpolate(prompt_guids[3], size=(7, 7), mode='bilinear', align_corners=True)
        sk2048 = SKAttention(channel=2048, reduction=8)
        sk2048.to(self.args.device)
        prompt_guids[3] = sk2048(prompt_guids[3])
        prompt_guids[3] = F.interpolate(prompt_guids[3], size=(2, 2), mode='bilinear', align_corners=True)
        # 2023-11-2修改SKAttention
        # aa = F.interpolate(prompt_guids[0] ,size=(4, 4), mode='bilinear')
        # prompt_guids[2] = F.interpolate(prompt_guids[2], size=(7, 7), mode='bilinear', align_corners=True)
        # se512 = SEAttention(channel=1024, reduction=8)
        # se512.to(self.args.device)
        # prompt_guids[2] = se512(prompt_guids[2])
        # prompt_guids[2] = F.interpolate(prompt_guids[2], size=(2, 2), mode='bilinear', align_corners=True)
        # 2023-10-31修改
        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840
        #这一行将主要的提示信息prompt_guids沿通道维度连接（dim=1），然后重新整形为形状[bsz, self.args.prompt_len, -1]，
        # 其中self.args.prompt_len是提示信息的长度。
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]
        #这一行对每个辅助提示信息aux_prompt_guid执行类似的连接和形状变换操作，将其存储在一个列表中。
        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        #这一行将主要提示信息prompt_guids通过卷积神经网络层self.encoder_conv进行处理，以生成进一步的表示。
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 x [bsz, 4, 4*2*768]
        #这一行对每个辅助提示信息 aux_prompt_guid 进行类似的处理，将其存储在一个列表中。
        split_prompt_guids = prompt_guids.split(768*2, dim=-1)   # 4 x [bsz, 4, 768*2]
        #这一行将主要提示信息 prompt_guids 沿通道维度分割为多个部分，每个部分的宽度为 768*2。
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4, 768*2]]
        #这一行对每个辅助提示信息 aux_prompt_guid 进行类似的分割操作，将其存储在一个列表中。
        result = []  #初始化一个空列表 result 用于存储最终结果。
        for idx in range(12): #这是一个循环，迭代次数为 12，表示为每个 Transformer 层执行以下操作。
            sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
            #这一行对主要提示信息的分割部分求和，然后除以 4，得到一个权重向量 sum_prompt_guids，用于加权主要提示信息的不同部分。
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)
            #这一行对 sum_prompt_guids 应用激活函数和 softmax 操作，生成表示提示信息的门控向量 prompt_gate。
            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
            #初始化一个与主要提示信息分割部分相同形状的零张量 key_val。
            for i in range(4):  #这是一个循环，迭代次数为 4，表示对主要提示信息的不同部分执行以下操作。
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])
            #这一行将加权后的主要提示信息与门控向量相乘并求和，生成 key_val，用于表示关键和值。
            aux_key_vals = []   # 3 x [bsz, 4, 768*2]
            #初始化一个空列表 aux_key_vals 用于存储辅助提示信息的关键和值。
            for split_aux_prompt_guid in split_aux_prompt_guids:  #这是一个循环，迭代处理不同的辅助提示信息。
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
                #这一行对辅助提示信息的分割部分求和，然后除以 4，得到一个权重向量 sum_aux_prompt_guids，用于加权不同部分的辅助提示信息。
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                #这一行对 sum_aux_prompt_guids 应用激活函数和 softmax 操作，生成表示辅助提示信息的门控向量 aux_prompt_gate。
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                #初始化一个与辅助提示信息分割部分相同形状的零张量 aux_key_val。
                for i in range(4):  #这是一个循环，迭代处理不同的辅助提示信息的不同部分。
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                    #这一行将加权后的辅助提示信息与门控向量相乘并求和，生成 aux_key_val，用于表示辅助提示信息的关键和值。
                aux_key_vals.append(aux_key_val)  #将每个辅助提示信息的关键和值 aux_key_val 存储在 aux_key_vals 列表中。
            key_val = [key_val] + aux_key_vals  #这一行将主要提示信息的关键和值与所有辅助提示信息的关键和值连接在一起，形成一个列表 key_val。
            key_val = torch.cat(key_val, dim=1)  #这一行将关键和值的列表 key_val 沿通道维度连接。
            key_val = key_val.split(768, dim=-1)  #这一行将连接的关键和值分割为 12 个部分，每个部分的宽度为 768。
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            #这一行重新组织关键和值的张量，以确保连续的内存布局。
            temp_dict = (key, value)  #创建一个包含关键和值的元组 temp_dict。
            result.append(temp_dict)  #将每个层的关键和值元组添加到 result 列表中。
        return result  #返回一个包含关键和值信息的列表 result 作为函数的输出。
class BertCrossEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers
class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output
class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if 768 % 12 != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (768, 12))
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / 12)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(768, self.all_head_size)
        self.key = nn.Linear(768, self.all_head_size)
        self.value = nn.Linear(768, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):

        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)



        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)



        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(768, 3072)
        if isinstance(gelu, str) or (sys.version_info[0] == 2 and isinstance(gelu, unicode)):
            self.intermediate_act_fn = ACT2FN[gelu]
        else:
            self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertPreTrainedModel1(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel1, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # Backward compatibility with old naming format
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model
class BertModel1(BertPreTrainedModel1):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertModel1, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.long)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

