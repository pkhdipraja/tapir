from dataset_readers.datasets import SeqLabellingDataModule, SeqLabellingRevisionDataModule
from model.model_module import LinearCausalEncoderLabelling, \
                               LinearEncoderLabelling, \
                               TransformerEncoderLabelling, \
                               IncrementalTransformerEncoderLabelling, \
                               TwoPassLabelling


model_dict = {
    'transformers_labelling': {'train': TransformerEncoderLabelling, 'test': TransformerEncoderLabelling},
    'linear-transformers_labelling': {'train': LinearEncoderLabelling, 'test': LinearEncoderLabelling},
    'linear-transformers-causal_labelling': {'train': LinearCausalEncoderLabelling, 'test': LinearEncoderLabelling},
    'incremental-transformers_labelling': {'train': IncrementalTransformerEncoderLabelling, 'test': IncrementalTransformerEncoderLabelling},
    'two-pass_labelling': {'train': TwoPassLabelling, 'test': TwoPassLabelling}
}
dm_dict = {
    'labelling': SeqLabellingDataModule,
    # 'classification': SeqClassificationDataModule,
    'labelling_revision': SeqLabellingRevisionDataModule
}
