import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import jinja2
import regex


GPT2_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class BatchEncoding(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def to(self, device: str):
        for key, value in self.items():
            if hasattr(value, "to"):
                self[key] = value.to(device)
        return self


class MyTokenizer:
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, tokenizer_path: Union[str, Path], config_path: Optional[Union[str, Path]] = None):
        tokenizer_path = Path(tokenizer_path)
        config_path = Path(config_path) if config_path else tokenizer_path.with_name("tokenizer_config.json")

        self.tokenizer_path = tokenizer_path
        self.config_path = config_path
        self.tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        self.config_data = json.loads(config_path.read_text(encoding="utf-8"))

        model_data = self.tokenizer_data["model"]
        self.vocab: Dict[str, int] = model_data["vocab"]
        self.id_to_token: Dict[int, str] = {idx: token for token, idx in self.vocab.items()}
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            tuple(merge): idx for idx, merge in enumerate(model_data["merges"])
        }
        self.cache: Dict[str, List[str]] = {}

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.added_tokens = self.tokenizer_data.get("added_tokens", [])
        self.added_token_map = {token["content"]: token for token in self.added_tokens}
        self.special_token_set = {
            token["content"] for token in self.added_tokens if token.get("special", False)
        }
        self.special_token_ids = {
            self.vocab[token] for token in self.special_token_set if token in self.vocab
        }
        self.non_special_added_token_set = {
            token["content"] for token in self.added_tokens if not token.get("special", False)
        }

        self.added_token_pattern = self._build_added_token_pattern(self.added_token_map.keys())

        self.bos_token = self.config_data["bos_token"]
        self.eos_token = self.config_data["eos_token"]
        self.pad_token = self.config_data["pad_token"]
        self.unk_token = self.config_data["unk_token"]
        self.additional_special_tokens = self.config_data.get("additional_special_tokens", [])
        self.clean_up_tokenization_spaces = self.config_data.get("clean_up_tokenization_spaces", False)
        self.spaces_between_special_tokens = self.config_data.get("spaces_between_special_tokens", False)
        self.chat_template = self.config_data.get("chat_template", "")

        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]

        self.special_tokens_map = {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "additional_special_tokens": self.additional_special_tokens,
        }
        self.all_special_tokens = self._dedupe_tokens(
            [self.bos_token, self.eos_token, self.unk_token, self.pad_token, *self.additional_special_tokens]
        )
        self.all_special_ids = [self.vocab[token] for token in self.all_special_tokens if token in self.vocab]

        self._jinja_env = jinja2.Environment(
            autoescape=False,
            trim_blocks=False,
            lstrip_blocks=False,
            keep_trailing_newline=True,
        )
        self._jinja_env.policies["json.dumps_kwargs"] = {
            "ensure_ascii": False,
            "sort_keys": False,
        }
        self._chat_template_compiled = self._jinja_env.from_string(self.chat_template) if self.chat_template else None

    @classmethod
    def from_pretrained(cls, path: Union[str, Path], *args, **kwargs):
        del args, kwargs
        path = Path(path)
        if path.is_dir():
            return cls(path / "tokenizer.json", path / "tokenizer_config.json")
        raise ValueError(f"Unsupported tokenizer path: {path}")

    def save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.tokenizer_path, save_directory / "tokenizer.json")
        shutil.copy2(self.config_path, save_directory / "tokenizer_config.json")
        return str(save_directory), str(save_directory / "tokenizer.json"), str(save_directory / "tokenizer_config.json")

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def convert_tokens_to_ids(self, tokens: Union[str, Sequence[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(
        self, ids: Union[int, Sequence[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            token = self.id_to_token.get(ids, self.unk_token)
            if skip_special_tokens and token in self.special_token_set:
                return ""
            return token

        tokens = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.unk_token)
            if skip_special_tokens and token in self.special_token_set:
                continue
            tokens.append(token)
        return tokens

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        open_thinking: bool = False,
        return_tensors: Optional[str] = None,
        **tokenizer_kwargs,
    ):
        if self._chat_template_compiled is None:
            raise ValueError("chat_template is not configured")

        rendered = self._chat_template_compiled.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            open_thinking=open_thinking,
        )
        if not tokenize:
            return rendered
        return self(rendered, add_special_tokens=False, return_tensors=return_tensors, **tokenizer_kwargs)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        return self._encode_text(text)

    def decode(
        self,
        token_ids: Union[Sequence[int], Any],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
    ) -> str:
        del clean_up_tokenization_spaces
        token_list = self._ensure_int_list(token_ids)
        pieces: List[str] = []
        for token_id in token_list:
            token = self.id_to_token.get(token_id, self.unk_token)
            if skip_special_tokens and token in self.special_token_set:
                continue
            pieces.append(token)
        return self._byte_level_decode("".join(pieces))

    def batch_decode(
        self,
        sequences: Sequence[Union[Sequence[int], Any]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
    ) -> List[str]:
        return [
            self.decode(
                sequence,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for sequence in sequences
        ]

    def __call__(
        self,
        text: Union[str, Sequence[str]],
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        del return_token_type_ids, kwargs

        is_batched = not isinstance(text, str)
        texts = list(text) if is_batched else [text]

        encoded = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]

        if padding is True:
            target_len = max(len(ids) for ids in encoded) if encoded else 0
        elif padding == "max_length":
            if max_length is None:
                raise ValueError("padding='max_length' requires max_length")
            target_len = max_length
        else:
            target_len = None

        attention_masks: List[List[int]] = []
        if target_len is not None:
            padded = []
            for ids in encoded:
                if truncation and len(ids) > target_len:
                    ids = ids[:target_len]
                pad_len = max(0, target_len - len(ids))
                padded.append(ids + [self.pad_token_id] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)
            encoded = padded
        elif return_attention_mask:
            attention_masks = [[1] * len(ids) for ids in encoded]

        if return_tensors is not None:
            if return_tensors != "pt":
                raise ValueError(f"Unsupported return_tensors={return_tensors!r}")
            import torch

            if target_len is None:
                max_len = max(len(ids) for ids in encoded) if encoded else 0
                padded = []
                tensor_attention_masks: List[List[int]] = []
                for ids in encoded:
                    pad_len = max_len - len(ids)
                    padded.append(ids + [self.pad_token_id] * pad_len)
                    if return_attention_mask:
                        tensor_attention_masks.append([1] * len(ids) + [0] * pad_len)
                encoded = padded
                if return_attention_mask:
                    attention_masks = tensor_attention_masks

            batch = BatchEncoding()
            batch["input_ids"] = torch.tensor(encoded, dtype=torch.long)
            if return_attention_mask:
                batch["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
            return batch

        batch = BatchEncoding()
        batch["input_ids"] = encoded if is_batched else encoded[0]
        if return_attention_mask:
            batch["attention_mask"] = attention_masks if is_batched else attention_masks[0]
        return batch

    def _encode_text(self, text: str) -> List[int]:
        token_ids: List[int] = []
        for is_added, piece in self._split_with_added_tokens(text):
            if not piece:
                continue
            if is_added:
                token_ids.append(self.vocab.get(piece, self.unk_token_id))
                continue

            for match in GPT2_PATTERN.findall(piece):
                transformed = "".join(self.byte_encoder[b] for b in match.encode("utf-8"))
                for bpe_token in self._bpe(transformed):
                    token_ids.append(self.vocab.get(bpe_token, self.unk_token_id))
        return token_ids

    def _split_with_added_tokens(self, text: str) -> List[Tuple[bool, str]]:
        if self.added_token_pattern is None:
            return [(False, text)]

        pieces: List[Tuple[bool, str]] = []
        last = 0
        for match in self.added_token_pattern.finditer(text):
            start, end = match.span()
            if start > last:
                pieces.append((False, text[last:start]))
            pieces.append((True, match.group(0)))
            last = end
        if last < len(text):
            pieces.append((False, text[last:]))
        return pieces

    def _bpe(self, token: str) -> List[str]:
        if token in self.cache:
            return self.cache[token]

        if token in self.vocab:
            self.cache[token] = [token]
            return [token]

        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            self.cache[token] = [token]
            return [token]

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break

                new_word.extend(word[i:j])
                i = j

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        result = list(word)
        self.cache[token] = result
        return result

    def _byte_level_decode(self, text: str) -> str:
        decoded = bytearray()
        for char in text:
            decoded.append(self.byte_decoder[char])
        return decoded.decode("utf-8", errors="replace")

    @staticmethod
    def _get_pairs(word: Sequence[str]) -> set:
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    @staticmethod
    def _build_added_token_pattern(tokens: Iterable[str]):
        tokens = list(tokens)
        if not tokens:
            return None
        tokens.sort(key=len, reverse=True)
        escaped = [regex.escape(token) for token in tokens]
        return regex.compile("|".join(escaped))

    @staticmethod
    def _ensure_int_list(token_ids: Union[Sequence[int], Any]) -> List[int]:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return [int(token_id) for token_id in token_ids]

    @staticmethod
    def _dedupe_tokens(tokens: Sequence[str]) -> List[str]:
        seen = set()
        deduped = []
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped
