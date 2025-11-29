import os
import numpy as np
from datasets import load_dataset
import tiktoken

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------

# Hugging Face 上の FineWeb-Edu データセット
REMOTE_NAME = "sample-10BT"  # 10B tokens サンプル
DATASET_NAME = "HuggingFaceFW/fineweb-edu"

# トークナイズ済みシャードの出力先
DATA_ROOT = "edu_fineweb10B"

# 1シャードあたりのトークン数（1e8 = 100M tokens）
SHARD_SIZE = int(1e8)

# ---------------------------------------------------------
# 準備
# ---------------------------------------------------------

os.makedirs(DATA_ROOT, exist_ok=True)

print(f"loading dataset: {DATASET_NAME} / {REMOTE_NAME}")
fw = load_dataset(DATASET_NAME, name=REMOTE_NAME, split="train")

enc = tiktoken.get_encoding("gpt2")
EOT_TOKEN = enc.eot_token  # 文末トークン


def tokenize(example):
    """1つのレコード(text)を uint16 のトークン配列に変換"""
    text = example["text"]
    ids = enc.encode_ordinary(text)
    ids.append(EOT_TOKEN)
    return np.array(ids, dtype=np.uint16)


def write_shard(shard_index, buf, token_count):
    """バッファの先頭 token_count 分を .npy に書き出し"""
    filename = os.path.join(DATA_ROOT, f"edufineweb_train_{shard_index:06d}.npy")
    np.save(filename, buf[:token_count])
    print(f"wrote {filename} ({token_count} tokens)")


def main():
    shard_index = 0
    token_buffer = np.empty(SHARD_SIZE, dtype=np.uint16)
    token_count = 0

    total_tokens = 0
    total_docs = 0

    for i, example in enumerate(fw):
        tokens = tokenize(example)
        n = len(tokens)

        # 今のバッファに入らないなら、一度書き出して新しいシャードを作る
        if token_count + n > SHARD_SIZE:
            write_shard(shard_index, token_buffer, token_count)
            shard_index += 1
            token_buffer = np.empty(SHARD_SIZE, dtype=np.uint16)
            token_count = 0

        token_buffer[token_count : token_count + n] = tokens
        token_count += n
        total_tokens += n
        total_docs += 1

        if (i + 1) % 1000 == 0:
            print(f"processed {i+1} docs, total_tokens={total_tokens}")

    # 端数のシャードを書き出し
    if token_count > 0:
        write_shard(shard_index, token_buffer, token_count)

    print("done.")
    print(f"total_docs={total_docs}, total_tokens={total_tokens}")


if __name__ == "__main__":
    main()
