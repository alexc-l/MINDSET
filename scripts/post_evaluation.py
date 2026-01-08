#!/usr/bin/env python3
# auto_two_step_aggregate.py
import json, re, argparse, numpy as np
from pathlib import Path
from collections import defaultdict

# -------------------- 配置 --------------------
COUNTRY_BLOCK = {
    # C1
    'Andorra': 'C1', 'Argentina': 'C1', 'Australia': 'C1', 'Bangladesh': 'C1',
    'Armenia': 'C1', 'Bolivia': 'C1', 'Brazil': 'C1', 'Myanmar': 'C1', 'Canada': 'C1',
    'Chile': 'C1', 'China': 'C1', 'Taiwan ROC': 'C1', 'Cyprus': 'C1', 'Ecuador': 'C1',
    'Germany': 'C1', 'Guatemala': 'C1', 'Hong Kong SAR': 'C1', 'Indonesia': 'C1',
    'Iran': 'C1', 'Iraq': 'C1', 'Japan': 'C1', 'Kazakhstan': 'C1', 'Jordan': 'C1',
    'South Korea': 'C1', 'Kyrgyzstan': 'C1', 'Lebanon': 'C1', 'Macao SAR': 'C1',
    'Maldives': 'C1', 'Mongolia': 'C1', 'Netherlands': 'C1', 'Nicaragua': 'C1',
    'Pakistan': 'C1', 'Philippines': 'C1', 'Romania': 'C1', 'Russia': 'C1',
    'Serbia': 'C1', 'Singapore': 'C1', 'Slovakia': 'C1', 'Vietnam': 'C1',
    'Tajikistan': 'C1', 'Turkey': 'C1', 'Ukraine': 'C1', 'Great Britain': 'C1',
    'United States': 'C1', 'Uruguay': 'C1', 'Venezuela': 'C1',

    # C2
    'Egypt': 'C2', 'Ethiopia': 'C2', 'Kenya': 'C2', 'Libya': 'C2',
    'Morocco': 'C2', 'Nigeria': 'C2', 'Tunisia': 'C2', 'Zimbabwe': 'C2',

    # C3
    'Malaysia': 'C3', 'Thailand': 'C3', 'Czechia': 'C3', 'Greece': 'C3',
    'Peru': 'C3', 'Colombia': 'C3', 'Mexico': 'C3', 'Puerto Rico': 'C3',
    'New Zealand': 'C3', "India": "C3"
}


def classify_question(q_key: str):
    nums = list(map(int, re.findall(r'\d+', q_key)))
    if not nums:
        return None
    n = nums[-1]
    if (1 <= n <= 130) or (152 <= n <= 175):
        return 'Q1'
    elif (131 <= n <= 151) or (176 <= n <= 198):
        return 'Q2'
    elif 199 <= n <= 259:
        return 'Q3'
    return None

def extract_country(folder_name: str) -> str:
    """
    第一个 '-' 与最后一个 '_' 之间为国家；
    若仍有 '_' → 替换成空格
    """
    # 去掉首尾空格
    name = folder_name.strip()
    # 第一个 '-'
    dash = name.find('-')
    if dash == -1:
        return name  # 没有 '-'，退化为原名字
    # 最后一个 '_'
    under = name.rfind('_')
    if under == -1 or under < dash:
        return name[dash+1:].replace('_', ' ')
    country = name[dash+1:under]
    return country.replace('_', ' ')

# -------------------- Step1：国家内聚合 --------------------
def step1_country_blocks(json_path: Path) -> Path:
    with open(json_path, encoding='utf-8') as f:
        q_metrics = json.load(f)
    bucket = {'Q1': [], 'Q2': [], 'Q3': []}
    for q_key, metrics in q_metrics.items():
        blk = classify_question(q_key)
        if blk:
            bucket[blk].append(metrics)
    result = {}
    for blk, arr in bucket.items():
        if not arr:
            continue
        result[blk] = {
            'avg_1_minus_jsd': float(np.mean([m['1_minus_jsd'] for m in arr])),
            'avg_emd': float(np.mean([m['emd'] for m in arr])),
            'avg_cohen_kappa': float(np.mean([m['cohen_kappa'] for m in arr])),
            'total_samples': int(np.sum([m['n_responses'] for m in arr]))
        }
    out_file = json_path.parent / 'country_blocks.json'
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return out_file


# -------------------- Step2：跨国家汇总 --------------------
def step2_final_aggregate(root: Path, output: Path):
    bucket = defaultdict(lambda: defaultdict(list))
    print('📊 Step2：扫到以下 country_blocks.json ：')
    for cb_file in root.rglob('country_blocks.json'):
        country = extract_country(cb_file.parent.name)
        print(f'  -> {country} ', end='')
        block = COUNTRY_BLOCK.get(country)
        if block is None:
            print('❌ 未匹配到 C1/C2/C3，跳过')
            continue
        print(f'✅ {block}')
        data = json.loads(cb_file.read_text(encoding='utf-8'))
        for q_blk, metrics in data.items():
            bucket[block][q_blk].append(metrics)

    if not bucket:
        print('⚠️  桶为空！没有有效国家-块数据，不写文件')
        return {}

    results = {}
    for block, q_dict in bucket.items():
        for q_blk, arr in q_dict.items():
            results[f'{block}-{q_blk}'] = {
                'block': block,
                'q_group': q_blk,
                'avg_1_minus_jsd': float(np.mean([m['avg_1_minus_jsd'] for m in arr])),
                'avg_emd': float(np.mean([m['avg_emd'] for m in arr])),
                'avg_cohen_kappa': float(np.mean([m['avg_cohen_kappa'] for m in arr])),
                'total_samples': int(np.sum([m['total_samples'] for m in arr]))
            }
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    return results


# -------------------- 一键入口 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='最上层目录（递归查找 question_level_metrics.json）')
    parser.add_argument('--output', default='final_blocks.json', help='最终汇总 json')
    args = argparse.Namespace(root=Path(parser.parse_args().root), output=Path(parser.parse_args().output))

    # 1. 自动 Step1：生成 country_blocks.json
    print('🔍 Step1：生成国家内 blocks …')
    for q_json in args.root.rglob('question_level_metrics.json'):
        step1_country_blocks(q_json)
    print('✅ Step1 完成')

    # 2. 自动 Step2：汇总所有 country_blocks.json
    print('📊 Step2：跨国家汇总 …')
    results = step2_final_aggregate(args.root, args.output)
    print('✅ Step2 完成')

    # 3. 终端打印表
    print('\n' + '='*60)
    print('📊 最终块级汇总')
    print('='*60)
    for k, v in results.items():
        print(f'{k:<10} | 1-JSD: {v["avg_1_minus_jsd"]:.4f} | EMD: {v["avg_emd"]:.4f} | κ: {v["avg_cohen_kappa"]:.4f} | N={v["total_samples"]}')
    print('='*60)

if __name__ == '__main__':
    main()