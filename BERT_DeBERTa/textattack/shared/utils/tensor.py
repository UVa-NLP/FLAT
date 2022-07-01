import numpy as np
import torch


def batch_model_predict(model_predict, inputs, batch_size=32):
    """Runs prediction on iterable ``inputs`` using batch size ``batch_size``.

    Aggregates all predictions into an ``np.ndarray``.
    """
    outputs = []
    i = 0
    while i < len(inputs):
        batch = inputs[i : i + batch_size]
        attention_mask = torch.zeros_like(batch)
        for ii in range(batch.shape[0]):
            mask_len = torch.count_nonzero(batch[ii])
            attention_mask[ii][:mask_len] = 1
        input = {'input_ids':      batch,
                   'attention_mask': attention_mask,
                   'token_type_ids': torch.zeros_like(batch),
                   'labels':         torch.ones(batch.shape[0], dtype=batch.dtype, device=batch.device)}
        try:
            out = model_predict(input_ids=input['input_ids'], attention_mask=input['attention_mask'],\
                token_type_ids=input['token_type_ids'], labels=input['labels'])
        except:
            out = model_predict(input)
        loss, logits = out[:2]
        batch_preds = logits

        # Some seq-to-seq models will return a single string as a prediction
        # for a single-string list. Wrap these in a list.
        if isinstance(batch_preds, str):
            batch_preds = [batch_preds]

        # Get PyTorch tensors off of other devices.
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu()

        # Cast all predictions iterables to ``np.ndarray`` types.
        if not isinstance(batch_preds, np.ndarray):
            # batch_preds = np.array(batch_preds)
            batch_preds = batch_preds.detach().numpy()
        outputs.append(batch_preds)
        i += batch_size

    return np.concatenate(outputs, axis=0)
