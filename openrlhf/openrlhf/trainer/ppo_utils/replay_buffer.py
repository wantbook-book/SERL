import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


from .experience_maker import Experience


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    base_action_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        vals = torch.unbind(v)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                assert vv.numel() == 1, f"info[{k}] must be a scalar tensor, but got {vv.shape}"
                vv = vv.item()
            batch_kwargs[i]["info"][k] = vv

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem], packing_samples=False) -> Experience:
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        vals = zero_pad_sequences(vals, "left") if vals[0] is not None else None
        kwargs[key] = vals

    kwargs["info"] = {}
    for key in items[0].info.keys():
        vals = torch.tensor([item.info[key] for item in items])
        kwargs["info"][key] = vals
    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, base_act_log_prob, value, ret, adv, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        )
        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
            base_act_log_prob[:right_pad] if item.base_action_log_probs is not None else None,
            value[:right_pad] if item.values is not None else None,
            ret[:right_pad],
            adv[:right_pad],
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
        )
    return items


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True, packing_samples: bool = False
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []
        self.sampled: List[bool] = []
        self.append_step: List[int] = []

    @torch.no_grad()
    def append(self, experience: Experience, step: int) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        self.items.extend(items)
        self.sampled.extend([False] * len(items))
        self.append_step.extend([step] * len(items))
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()
    
    def clear_rule(self, step: int, n_samples_per_prompt: int) -> None:
        remove_indices = []
        expired_indices = []
        for idx, (sampled, append_step) in enumerate(zip(self.sampled, self.append_step)):
            # fixed 3
            # if sampled or step - append_step >= 3:
            if sampled:
                remove_indices.append(idx)
            elif step - append_step >= 3:
                expired_indices.append(idx)
        expired_indices = expired_indices[: len(expired_indices) // n_samples_per_prompt * n_samples_per_prompt]
        remove_indices.extend(expired_indices)
        remove_indices = sorted(remove_indices, reverse=True)
        
        for idx in remove_indices:
            del self.items[idx]
            del self.sampled[idx]
            del self.append_step[idx]

    @torch.no_grad()
    def sample(self) -> Experience:
        # items = random.sample(self.items, self.sample_batch_size)
        indices = random.sample(range(len(self.items)), self.sample_batch_size)
        items = []
        for idx in indices:
            # self.sampled[idx] = True
            items.append(self.items[idx])
            # self.items[idx].sampled = True

        experience = make_experience_batch(items, self.packing_samples)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        self.sampled[idx] = True
        # print(f"sampled {idx} with {self.sampled[idx]}")
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch, self.packing_samples)
        return experience
