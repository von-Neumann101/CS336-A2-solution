import argparse
from statistics import mean, pstdev
from timeit import default_timer

import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


# 每组数字依次是 d_model、d_ff、层数、注意力头数。
MODEL_SIZES = {
    "toy":    (64, 256, 2, 4),
    "small":  (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large":  (1280, 5120, 36, 20),
    "xl":     (2560, 10240, 32, 32),
    "10b":    (4608, 12288, 50, 36),
}


def run_step(model, optimizer, x, targets, mode):
    if mode == "forward":
        with torch.no_grad():
            model(x)
        return

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = cross_entropy(logits, targets)
    loss.backward()

    if mode == "train":
        optimizer.step()


def main():
    # 读取命令行参数
    parser = argparse.ArgumentParser() # 命令行参数解析器
    parser.add_argument("--size", choices=MODEL_SIZES, default="toy")
    parser.add_argument(
        "--mode",
        choices=["forward", "forward-backward", "train"],
        default="train",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1 or args.seq_len < 1:
        parser.error("warmup must be nonnegative, steps and seq-len must be positive")

    torch.manual_seed(0) # 把所有的默认随机数生成的种子设置为 0
    vocab_size = 10_000 # 10_000便于阅读，等于10000
    d_model, d_ff, num_layers, num_heads = MODEL_SIZES[args.size]

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=args.seq_len,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to("cuda")

    model.train(args.mode != "forward") # 传入 False 将模型切换为 eval 模式
    optimizer = AdamW(model.parameters())
    # 4 是 batch_size，这一行向 GPU 提交任务
    tokens = torch.randint(vocab_size, (4, args.seq_len + 1), device="cuda")
    x = tokens[:, :-1]
    targets = tokens[:, 1:]
    torch.cuda.synchronize() # 保证 GPU 处理完任务才进行下面的操作，保证计时准确
    # 如果没有synchronize，GPU 还在处理随机数，但是 CPU 已经开始执行下面的语句了
    
    times_ms = []
    for step in range(args.warmup + args.steps):
        start = default_timer()
        run_step(model, optimizer, x, targets, args.mode)
        torch.cuda.synchronize()
        elapsed_ms = (default_timer() - start) * 1000

        if step >= args.warmup: 
            # warmup 是预热时间，保证 GPU 加载了全部东西，这样测量的就是真正运算的时间
            times_ms.append(elapsed_ms)

    print(f"每步耗时（ms）: {[round(t, 2) for t in times_ms]}")
    print(f"均值 {mean(times_ms):.2f} ms，标准差 {pstdev(times_ms):.2f} ms")


if __name__ == "__main__":
    main()