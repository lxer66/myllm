# MiniMind Tokenizer 复刻说明

## 目标

我们的目标不是重新训练一个新的 tokenizer，而是：

- 自己实现一个纯 Python tokenizer
- 读取仓库现有的 [model/tokenizer.json](/d:/STUDY/course/6/NLP/minimind/model/tokenizer.json) 和 [model/tokenizer_config.json](/d:/STUDY/course/6/NLP/minimind/model/tokenizer_config.json)
- 尽量复刻 Hugging Face 当前对这两个文件的处理逻辑
- 除了 tokenizer 的实现方式不同，其余训练、推理、评测链路都保持和仓库一致

这意味着我们真正要复刻的不是“某个抽象概念上的 BPE”，而是：

- `transformers.AutoTokenizer.from_pretrained("model")`
- 在加载这两个本地文件后，实际表现出的编码、解码、chat template、special token 处理行为

## 先说结论

就这个仓库而言，Hugging Face 的 tokenizer 逻辑并不算特别复杂，原因是本地 `tokenizer.json` 结构比较干净：

- `normalizer = None`
- `pre_tokenizer = ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)`
- `model = BPE`
- `post_processor = None`
- `decoder = ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)`

也就是说，这个仓库的 tokenizer 主干逻辑可以概括成：

1. 用 `tokenizer_config.json` 中的 Jinja `chat_template` 先把消息渲染成字符串
2. 对原始文本优先匹配 `added_tokens`
3. 剩余普通文本再走 `ByteLevel pre-tokenizer`
4. 对预处理后的片段做 `BPE merge`
5. 映射成 token id
6. 解码时，只有 `special=True` 的 token 会在 `skip_special_tokens=True` 时被跳过

这个结论不是拍脑袋得出的，而是结合了：

- 仓库本地 `tokenizer.json` / `tokenizer_config.json` 结构
- Hugging Face 官方文档
- 本地实际调用 `AutoTokenizer` 的行为核验

## 一、这个仓库里 HF 实际在做什么

### 1. 加载哪些文件

仓库中几乎所有入口都是通过 `AutoTokenizer.from_pretrained(...)` 加载 tokenizer，例如：

- [trainer/trainer_utils.py](/d:/STUDY/course/6/NLP/minimind/trainer/trainer_utils.py:119)
- [eval_llm.py](/d:/STUDY/course/6/NLP/minimind/eval_llm.py:13)
- [scripts/serve_openai_api.py](/d:/STUDY/course/6/NLP/minimind/scripts/serve_openai_api.py:29)

对当前 `model/` 目录来说，HF 主要读取的是：

- [model/tokenizer.json](/d:/STUDY/course/6/NLP/minimind/model/tokenizer.json)
- [model/tokenizer_config.json](/d:/STUDY/course/6/NLP/minimind/model/tokenizer_config.json)

其中：

- `tokenizer.json` 存的是分词规则本体
- `tokenizer_config.json` 存的是 tokenizer 的外围配置和 `chat_template`

### 2. chat template 先发生

在对话训练和推理链路里，很多地方先调用：

- `tokenizer.apply_chat_template(...)`

例如：

- [dataset/lm_dataset.py](/d:/STUDY/course/6/NLP/minimind/dataset/lm_dataset.py:81)
- [eval_llm.py](/d:/STUDY/course/6/NLP/minimind/eval_llm.py:76)
- [scripts/serve_openai_api.py](/d:/STUDY/course/6/NLP/minimind/scripts/serve_openai_api.py:107)

这说明 `apply_chat_template` 不是“可有可无的附加功能”，而是输入构造的一部分。

对这个仓库来说，正确顺序是：

1. 先用 `chat_template` 把 `messages/tools/open_thinking/add_generation_prompt` 渲染成字符串
2. 再把这个字符串送入 tokenizer 编码

不要反过来做，也不要手写一个“差不多”的模板替代原模板。

### 3. added tokens 会先整体匹配

这是最容易误解的一点。

HF 并不是“先把整段文本当普通文本跑 ByteLevel + BPE，最后再想办法识别特殊串”。  
更接近实际的说法是：

1. tokenizer 中已经注册了一批 `added_tokens`
2. 这些 token 在编码时会被优先整体识别
3. 没命中的普通文本片段才会进入后续的 ByteLevel + BPE 流程

本仓库里，本地 `tokenizer.json` 显示：

- 一共有 36 个 `added_tokens`
- 其中 21 个 `special=True`
- 15 个 `special=False`

并且它们的匹配参数都相同：

- `single_word=False`
- `lstrip=False`
- `rstrip=False`
- `normalized=False`

这意味着在这个仓库里，`added_tokens` 的处理逻辑可以简化理解为：

- 按原始文本精确匹配
- 不要求词边界
- 不吞左空格
- 不吞右空格
- 不做额外规范化
- 一旦命中，就整体作为一个 token 处理

### 4. 剩余普通文本才走 ByteLevel + BPE

对于没有被 `added_tokens` 命中的普通文本，才会进入：

- `ByteLevel pre-tokenizer`
- `BPE model`

本仓库的配置是：

- `pre_tokenizer = ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)`
- `model = BPE`
- `merges_size = 6108`
- `vocab_size = 6400`

所以复刻时不能偷懒写成“按字符切分后直接查 vocab”，也不能省略 ByteLevel 这一步。

这个仓库里，ByteLevel 的典型表现是：

- 空格相关 token 会以类似 `\u0120` 前缀的形式存在于词表语义中
- 换行 `\n` 会映射成 `\u010a`
- 回车 `\r` 会映射成 `\u010d`

本地实测结果也验证了这一点，例如：

- `"a b"` 会编码为 `['a', '\\u0120b']`
- `"a  b"` 会编码为 `['a', '\\u0120', '\\u0120b']`
- `"\n"` 会编码为 `['\\u010a']`
- `"\r\n"` 会编码为 `['\\u010d', '\\u010a']`

### 5. 这个仓库没有 post-processor，也不会自动加 bos/eos

这一点非常重要。

从本地文件可知：

- `tokenizer.json` 中 `post_processor = None`
- `tokenizer_config.json` 中 `add_bos_token = false`
- `tokenizer_config.json` 中 `add_eos_token = false`

所以当前 tokenizer 本身不会自动在普通编码时额外添加：

- `bos_token`
- `eos_token`

也就是说，这个仓库的 `<|im_start|>` / `<|im_end|>` 主要是靠：

- chat template
- 数据集代码
- 上层逻辑

显式拼进输入文本里的，而不是 tokenizer 自动注入的。

### 6. decode 时，只有 special token 会被 skip

这是另一个必须说清楚的点。

HF 在 `skip_special_tokens=True` 时，不是“删除所有 added token”，而是只删除：

- 被标记为 `special=True` 的 token

对这个仓库，本地实测已经确认：

- `<think>` 编码后是单独一个 token，且 `skip_special_tokens=True` 时不会被删
- `</think>` 同理
- `<tool_call>...</tool_call>` 里的标签也不会因为 `skip_special_tokens=True` 被删
- `<|im_start|>` 和 `<|im_end|>` 会被删

这和本地 `tokenizer.json` 的定义完全一致：

- `<|im_start|>`、`<|im_end|>` 是 `special=True`
- `<think>`、`</think>`、`<tool_call>`、`</tool_call>`、`<tool_response>`、`</tool_response>` 是 `special=False`

因此，复刻时必须对齐这条规则：

- `added token` 不等于 `special token`
- 所有 added token 都要整体匹配
- 但只有 `special=True` 的 added token 才会在 `skip_special_tokens=True` 时被过滤

## 二、为什么 special token / added token 很关键

不是因为这些 token “是专门训练给 HF 用的”，而是因为：

- 它们在本地 json 里不只是“一个字符串 + 一个 id”
- 它们还有“如何被匹配、何时整体保留、何时被 skip”的运行语义

这套语义 HF 已经实现好了，而我们自己实现时必须显式复刻。

### 1. 训练链路依赖这些边界

例如 [dataset/lm_dataset.py](/d:/STUDY/course/6/NLP/minimind/dataset/lm_dataset.py:65) 和 [dataset/lm_dataset.py](/d:/STUDY/course/6/NLP/minimind/dataset/lm_dataset.py:81) 会根据：

- `<|im_start|>assistant\n`
- `<|im_end|>\n`

对应的 token 序列来定位 assistant span，并计算 label mask。

这意味着如果我们把这些 token 的行为弄错，后果不只是：

- token id 变了

还可能是：

- assistant 区间识别错了
- loss mask 算错了
- 训练目标发生偏移

### 2. tool call / think 标签不是普通装饰

仓库中：

- `<think>` / `</think>`
- `<tool_call>` / `</tool_call>`
- `<tool_response>` / `</tool_response>`

会直接参与：

- 训练数据构造
- agent rollout
- 推理响应解析

例如：

- [trainer/train_agent.py](/d:/STUDY/course/6/NLP/minimind/trainer/train_agent.py:107)
- [scripts/serve_openai_api.py](/d:/STUDY/course/6/NLP/minimind/scripts/serve_openai_api.py:83)

所以这些标签必须：

- 编码时整体匹配
- decode 时稳定保留
- 不能误判为需要 skip 的 special token

## 三、这个仓库里必须重点对齐的 token

### 1. 真正的 special tokens

这类 token 需要：

- 整体匹配
- 保留固定 id
- `skip_special_tokens=True` 时会被跳过

当前最关键的是：

- `<|endoftext|>`
- `<|im_start|>`
- `<|im_end|>`

以及视觉/音频相关 token：

- `<|vision_start|>`
- `<|vision_end|>`
- `<|vision_pad|>`
- `<|image_pad|>`
- `<|video_pad|>`
- `<|audio_start|>`
- `<|audio_end|>`
- `<|audio_pad|>`
- `<tts_pad>`
- `<tts_text_bos>`
- `<tts_text_eod>`
- `<tts_text_bos_single>`

### 2. added 但非 special 的 tokens

这类 token 同样需要：

- 整体匹配
- 保留固定 id

但它们在 `skip_special_tokens=True` 时不能被删掉。

当前最关键的是：

- `<tool_call>`
- `</tool_call>`
- `<tool_response>`
- `</tool_response>`
- `<think>`
- `</think>`

此外还包括：

- `<|buffer1|>` 到后续 buffer token

## 四、对这个仓库来说，正确的处理顺序是什么

下面这份流程是目前最适合复刻的“准实现说明”。

### 1. `apply_chat_template`

输入：

- `messages`
- `tools`
- `add_generation_prompt`
- `open_thinking`

处理：

- 直接执行 `tokenizer_config.json` 中保存的 `chat_template` Jinja 模板
- 得到最终字符串

注意：

- 不要手写一个“自己理解的 chat template”
- 不要随意改换行、空格、空字符串处理逻辑

### 2. `encode(text, add_special_tokens=False)`

处理：

1. 在原始文本上扫描 `added_tokens`
2. 命中的 added token 直接作为单独片段保留
3. 未命中的普通文本片段才进入 `ByteLevel pre-tokenizer`
4. 再对 ByteLevel 结果做 BPE merge
5. 最后映射成 token ids

注意：

- 这里的“先扫描 added token”是必须的
- 不能先把全文都送去普通 BPE，再试图回头识别 `<think>`
- `normalized=False` 表示按原始字符串匹配，不做额外规范化
- `single_word=False` 表示不要求词边界
- `lstrip=False`、`rstrip=False` 表示不吃掉左右空白

### 3. `decode(ids, skip_special_tokens=False)`

处理：

1. `id -> token string`
2. 如果 `skip_special_tokens=True`，只过滤 `special=True` 的 token
3. 保留所有非 special 的 added token
4. 对剩余 token 串做 `ByteLevel decoder`
5. 输出文本

注意：

- `<think>` 不能因为 `skip_special_tokens=True` 被删
- `<tool_call>` 也不能被删
- `<|im_start|>`、`<|im_end|>` 可以被删

## 五、哪些地方不应该自己“脑补优化”

为了尽量和 HF 对齐，下面这些地方不要自作主张改动：

- 不要新增 normalizer
- 不要自动加 bos/eos
- 不要把所有 added token 都当成 special token
- 不要把 `<think>` / `<tool_call>` 当普通文本拆开
- 不要自己简化 ByteLevel
- 不要自己重写 chat template 的拼接逻辑
- 不要擅自清理空格或换行

## 六、实现时建议暴露的最小接口

为了尽量兼容仓库现有代码，建议至少实现：

- `encode`
- `decode`
- `batch_decode`
- `__call__`
- `apply_chat_template`
- `convert_ids_to_tokens`
- `bos_token`
- `eos_token`
- `pad_token`
- `unk_token`
- `bos_token_id`
- `eos_token_id`
- `pad_token_id`
- `unk_token_id`

`__call__` 最少要支持这些常见参数：

- `add_special_tokens`
- `truncation`
- `max_length`
- `padding`
- `return_tensors="pt"`

因为仓库里训练和推理脚本都会直接这样用。

## 七、必须做的一致性测试

因为目标是“除了 tokenizer 的实现方式不同，其他都和仓库一模一样”，所以不能只看代码“像不像”，必须把 HF 当参考裁判做对照测试。

建议至少做下面这些测试。

### 1. 普通文本编码对齐

测试：

- 中文
- 英文
- 中英混合
- JSON 片段
- XML/标签片段
- 空格、连续空格、换行、回车换行、制表符

检查：

- `encode(..., add_special_tokens=False)` 的 `input_ids` 是否完全一致

### 2. added token 整体匹配

测试：

- `<think>`
- `</think>`
- `<tool_call>{"name":"x"}</tool_call>`
- `<|im_start|>assistant\n你好<|im_end|>\n`

检查：

- 对这些串编码后，关键标签是否始终作为单个 token 出现

### 3. decode 对齐

检查：

- `decode(ids, skip_special_tokens=False)` 是否一致
- `decode(ids, skip_special_tokens=True)` 是否一致

特别检查：

- `<think>` 不应在 `skip_special_tokens=True` 时消失
- `<|im_start|>` / `<|im_end|>` 应该会被跳过

### 4. chat template 对齐

对同一组：

- `messages`
- `tools`
- `add_generation_prompt`
- `open_thinking`

检查：

- `apply_chat_template(..., tokenize=False)` 输出字符串是否完全一致

### 5. 数据集端到端对齐

直接拿仓库现有数据集流程，例如：

- [dataset/lm_dataset.py](/d:/STUDY/course/6/NLP/minimind/dataset/lm_dataset.py)

检查：

- 同一样本的 `input_ids` 是否一致
- 同一样本的 `labels` / loss mask 是否一致

这是最关键的最终验收。

## 八、对当前作业场景的建议

如果作业要求是“自己实现 tokenizer”，那最稳的路线就是：

1. 不重新训练 tokenizer
2. 直接读取现有 `tokenizer.json` 和 `tokenizer_config.json`
3. 自己实现 tokenizer 执行逻辑
4. 用 HF tokenizer 做对照测试
5. 全通过后，再在训练脚本里替换成自研实现

这样做的好处是：

- 满足“自己实现 tokenizer”的作业要求
- 尽量减少和原仓库行为偏差
- 避免额外引入一个新的 tokenizer 分支

## 九、给 AI 的实现 Prompt

下面这段 Prompt 可以直接给另一个 AI，用来生成实现代码。最好配合当前仓库的 `model/tokenizer.json` 和 `model/tokenizer_config.json` 一起使用。

```text
请你为 MiniMind 仓库实现一个纯 Python tokenizer，目标不是重新训练 tokenizer，而是严格读取现有的 model/tokenizer.json 和 model/tokenizer_config.json，尽量复刻 Hugging Face AutoTokenizer.from_pretrained("model") 的行为。

要求如下：

1. 总目标
- 我们的目标是：除了 tokenizer 的实现方式不同，其他训练、推理、评测链路都尽量和当前仓库一致。
- 不要重新设计 tokenizer，不要重新训练词表。
- 只实现“读取现有 json 并执行其规则”的 tokenizer。

2. 必须读取的文件
- model/tokenizer.json
- model/tokenizer_config.json

3. 这个仓库的 tokenizer 结构
- normalizer = None
- pre_tokenizer = ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
- model = BPE
- post_processor = None
- decoder = ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)
- add_bos_token = false
- add_eos_token = false
- chat_template 在 tokenizer_config.json 中

4. apply_chat_template 的行为
- 不要手写一个“差不多”的聊天模板。
- 直接执行 tokenizer_config.json 中保存的 chat_template（Jinja 模板）。
- 至少支持 messages、tools、add_generation_prompt、open_thinking 这些参数。
- tokenize=False 时返回字符串。

5. added token 的处理逻辑
- tokenizer.json 中的 added_tokens 必须优先整体匹配，不能被普通 ByteLevel/BPE 拆开。
- added_tokens 的匹配发生在 ByteLevel + BPE 之前。
- 这个仓库里所有 added_tokens 的参数都是：
  - single_word=False
  - lstrip=False
  - rstrip=False
  - normalized=False
- 因此匹配时：
  - 按原始文本精确匹配
  - 不要求词边界
  - 不吞左空格
  - 不吞右空格
  - 不做额外规范化

6. special token 和 added token 的区别
- added token 不等于 special token。
- 所有 added token 都要整体匹配。
- 但 decode(..., skip_special_tokens=True) 时，只跳过 special=True 的 token。
- 不要跳过 special=False 的 added token。
- 这个仓库中，以下 token 必须整体匹配，但不能在 skip_special_tokens=True 时被删除：
  - <think>
  - </think>
  - <tool_call>
  - </tool_call>
  - <tool_response>
  - </tool_response>
- 这个仓库中，以下 token 是 special=True，skip_special_tokens=True 时应该被跳过：
  - <|endoftext|>
  - <|im_start|>
  - <|im_end|>
  - 以及 tokenizer.json 中定义的其他 special=True 的 added token

7. encode 主流程
- 先在原始文本中扫描并切出 added_tokens
- 未命中的普通文本片段再走 ByteLevel pre-tokenizer
- 然后做 BPE merge
- 再映射到 id
- 不要自动添加 bos/eos，因为 add_bos_token/add_eos_token 都是 false，且 post_processor 为 None

8. decode 主流程
- id -> token string
- skip_special_tokens=True 时只过滤 special=True 的 token
- 保留非 special 的 added token
- 对剩余 token 串做 ByteLevel decoder
- 返回文本

9. 必须实现的接口和属性
- encode
- decode
- batch_decode
- __call__
- apply_chat_template
- convert_ids_to_tokens
- bos_token / eos_token / pad_token / unk_token
- bos_token_id / eos_token_id / pad_token_id / unk_token_id

10. __call__ 最少支持的参数
- add_special_tokens
- truncation
- max_length
- padding
- return_tensors="pt"

11. 一致性测试
请同时编写测试脚本，对照 Hugging Face AutoTokenizer.from_pretrained("model") 检查以下内容是否完全一致：
- 普通文本 encode 结果一致
- decode 结果一致
- apply_chat_template(..., tokenize=False) 一致
- <think> / <tool_call> 等 added token 不被拆分
- skip_special_tokens=True 时，仅 special=True 的 token 被跳过
- 用 dataset/lm_dataset.py 中的样本跑出来的 input_ids 和 labels 结果一致

12. 实现原则
- 不要为“简化实现”而省略 ByteLevel
- 不要自行发明新规则
- 不要把所有 added token 都当 special token
- 不要自动清理空格或换行
- 优先保证行为和 HF 一致，不需要追求性能
```
