"""SFT 模型对话测试。"""
import sys, os, time, random, argparse, warnings, torch
from transformers import TextStreamer

__package__ = "homework"
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from model.model import MicroLMForCausalLM, Config
from model.tokenizer import MyTokenizer
from trainer.trainer_utils import setup_seed
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description="MicroLM SFT 模型对话")
    parser.add_argument('--weight_path', default='out/full_sft_768.pth', type=str)
    parser.add_argument('--tokenizer_path', default='../model', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--max_new_tokens', default=512, type=int)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（0=不携带，需为偶数）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    args = parser.parse_args()

    print(f"加载模型: {args.weight_path}")
    config = Config(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    model = MicroLMForCausalLM(config)
    model.load_state_dict(torch.load(args.weight_path, map_location=args.device), strict=True)
    model = model.half().eval().to(args.device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total:.2f}M")

    print(f"加载 tokenizer: {args.tokenizer_path}")
    tokenizer = MyTokenizer.from_pretrained(args.tokenizer_path)

    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '解释什么是机器学习',
        '推荐一些中国的美食',
    ]

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    conversation = []

    while True:
        prompt = input('\n💬 请输入（直接回车使用测试提示词，q 退出）: ').strip()
        if prompt == 'q':
            break

        test_mode = prompt == ''
        if test_mode:
            prompt = random.choice(prompts)
            print(f'💬: {prompt}')

        setup_seed(random.randint(0, 31415926))

        # 对话模式：使用 chat template
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        inputs_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='', flush=True)
        st = time.time()
        generated = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
        response = tokenizer.decode(
            generated[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": response})

        gen_tokens = len(generated[0]) - len(inputs["input_ids"][0])
        speed = gen_tokens / max(time.time() - st, 0.001)
        print(f'\n[速度: {speed:.1f} tokens/s, 生成 {gen_tokens} tokens]')


if __name__ == "__main__":
    main()
