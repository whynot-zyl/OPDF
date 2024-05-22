from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch.nn as nn

class BertModelCustom(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重（非常重要）
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=True
        )

        # 用于分类的隐藏层状态
        pooled_output = outputs[1]

        # 应用分类器
        logits = self.classifier(pooled_output)

        # 如果提供了标签，计算并返回损失；否则，只返回logits
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits, outputs.hidden_states

        return logits, outputs.hidden_states

