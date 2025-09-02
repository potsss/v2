#!/usr/bin/env python3
"""
用户扩量使用示例
演示如何使用用户扩量系统进行Look-alike建模
"""

import os
import pandas as pd
import numpy as np
from user_expansion import UserExpansionDataset, UserExpansionModel


def create_sample_seed_users(embeddings_path: str, output_path: str, sample_ratio: float = 0.01):
    """
    创建示例种子用户文件
    从用户向量文件中随机采样一部分用户作为种子用户
    
    Args:
        embeddings_path: 用户向量文件路径
        output_path: 种子用户文件输出路径
        sample_ratio: 采样比例
    """
    import pickle
    
    print(f"从 {embeddings_path} 创建示例种子用户...")
    
    # 加载用户向量
    with open(embeddings_path, 'rb') as f:
        user_embeddings = pickle.load(f)
    
    # 随机采样
    all_users = list(user_embeddings.keys())
    sample_size = max(1, int(len(all_users) * sample_ratio))
    seed_users = np.random.choice(all_users, size=sample_size, replace=False)
    
    # 保存为CSV文件
    seed_df = pd.DataFrame({'user_id': seed_users})
    seed_df.to_csv(output_path, index=False)
    
    print(f"创建了 {len(seed_users)} 个种子用户，保存到: {output_path}")
    return output_path


def run_expansion_example():
    """运行扩量示例"""
    
    # 假设的文件路径（需要根据实际情况修改）
    embeddings_path = "experiments/location_feature/models/comprehensive_user_embeddings.pkl"
    seed_users_path = "sample_seed_users.csv"
    output_dir = "expansion_results"
    
    print("=== 用户扩量示例 ===\n")
    
    # 检查文件是否存在
    if not os.path.exists(embeddings_path):
        print(f"错误: 用户向量文件不存在: {embeddings_path}")
        print("请先运行以下命令生成用户向量:")
        print("python main.py --mode compute_fused_embeddings --enable_attributes --enable_location")
        return
    
    # 1. 创建示例种子用户（如果不存在）
    if not os.path.exists(seed_users_path):
        print("1. 创建示例种子用户...")
        create_sample_seed_users(embeddings_path, seed_users_path, sample_ratio=0.05)
    else:
        print("1. 使用现有种子用户文件...")
    
    # 2. 加载数据
    print("\n2. 加载数据...")
    dataset = UserExpansionDataset(embeddings_path, seed_users_path)
    dataset.load_data()
    dataset.prepare_features(use_feature_engineering=True)
    
    # 3. 数据平衡
    print("\n3. 平衡数据集...")
    dataset.balance_dataset(strategy='undersample', ratio=5)
    
    # 4. 划分数据集
    print("\n4. 划分训练和验证集...")
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        dataset.features, dataset.labels,
        test_size=0.2, stratify=dataset.labels,
        random_state=42
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    
    # 5. 训练模型
    print("\n5. 训练模型...")
    model = UserExpansionModel(random_state=42)
    
    # 训练传统机器学习模型
    results = model.train_traditional_models(X_train, y_train, X_val, y_val)
    
    # 显示模型性能
    print("\n模型性能对比:")
    for name, result in results.items():
        print(f"{name:20s} - AUC: {result['auc']:.4f}, F1: {result['f1']:.4f}")
    
    # 6. 预测所有用户
    print("\n6. 预测所有用户...")
    predictions_df = model.predict_all_users(dataset)
    
    # 7. 获取扩量用户
    print("\n7. 生成扩量结果...")
    top_k = min(1000, len(predictions_df[predictions_df['is_seed'] == 0]))
    expansion_users = predictions_df[predictions_df['is_seed'] == 0].head(top_k)
    
    # 8. 计算群体相似度
    print("\n8. 计算群体相似度...")
    seed_user_list = list(dataset.seed_users)
    top_user_list = expansion_users['user_id'].tolist()
    
    group_similarities = model.compute_group_similarities(dataset, top_user_list, seed_user_list)
    
    # 9. 保存结果
    print("\n9. 保存结果...")
    os.makedirs(output_dir, exist_ok=True)
    
    predictions_df.to_csv(os.path.join(output_dir, "all_users_predictions.csv"), index=False)
    expansion_users.to_csv(os.path.join(output_dir, "expansion_results.csv"), index=False)
    
    # 保存群体相似度结果
    group_similarities_df = pd.DataFrame([group_similarities])
    group_similarities_df.to_csv(os.path.join(output_dir, "group_similarities.csv"), index=False)
    
    model.save_model(os.path.join(output_dir, "expansion_model.pkl"))
    
    # 11. 结果分析
    print("\n=== 扩量结果分析 ===")
    print(f"总用户数: {len(predictions_df):,}")
    print(f"种子用户数: {len(dataset.seed_users):,}")
    print(f"扩量用户数: {len(expansion_users):,}")
    print(f"扩量率: {len(expansion_users) / len(dataset.seed_users):.1f}x")
    
    print(f"\n扩量分数统计:")
    print(f"平均分数: {expansion_users['score'].mean():.4f}")
    print(f"分数标准差: {expansion_users['score'].std():.4f}")
    print(f"最高分数: {expansion_users['score'].max():.4f}")
    print(f"最低分数: {expansion_users['score'].min():.4f}")
    
    # 群体相似度统计
    print(f"\n=== 群体相似度统计 ===")
    for emb_type in ['fused_embedding', 'behavior_embedding', 'location_embedding', 'attribute_embedding']:
        sim_key = f'{emb_type}_group_similarity'
        if sim_key in group_similarities and group_similarities[sim_key] is not None:
            print(f"{emb_type:20s}: {group_similarities[sim_key]:.4f}")
    
    # 12. 显示top-10结果
    print(f"\nTop-10 扩量用户:")
    print(expansion_users[['user_id', 'score']].head(10).to_string(index=False))
    
    print(f"\n结果已保存到目录: {output_dir}")
    print("文件说明:")
    print("- all_users_predictions.csv: 所有用户的预测分数")
    print("- expansion_results.csv: Top-K扩量用户")
    print("- group_similarities.csv: 群体相似度结果")
    print("- expansion_model.pkl: 训练好的扩量模型")


def analyze_results(results_path: str = "expansion_results/expansion_results.csv", 
                   group_similarities_path: str = "expansion_results/group_similarities.csv"):
    """分析扩量结果"""
    
    if not os.path.exists(results_path):
        print(f"结果文件不存在: {results_path}")
        return
    
    print("=== 扩量结果深度分析 ===\n")
    
    # 加载扩量结果
    results_df = pd.read_csv(results_path)
    print(f"扩量用户总数: {len(results_df)}")
    
    # 分数分布分析
    print(f"\n分数分布:")
    print(f"均值: {results_df['score'].mean():.4f}")
    print(f"中位数: {results_df['score'].median():.4f}")
    print(f"标准差: {results_df['score'].std():.4f}")
    
    # 分数分位数
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    print(f"\n分数分位数:")
    for q in quantiles:
        print(f"P{int(q*100):2d}: {results_df['score'].quantile(q):.4f}")
    
    # 高质量扩量用户
    high_score_threshold = results_df['score'].quantile(0.9)
    high_score_users = results_df[results_df['score'] >= high_score_threshold]
    
    print(f"\n高分用户分析 (分数 >= {high_score_threshold:.4f}):")
    print(f"高分用户数: {len(high_score_users)}")
    
    # 群体相似度分析
    if os.path.exists(group_similarities_path):
        print(f"\n=== 群体相似度分析 ===")
        group_sim_df = pd.read_csv(group_similarities_path)
        
        for emb_type in ['fused_embedding', 'behavior_embedding', 'location_embedding', 'attribute_embedding']:
            sim_col = f'{emb_type}_group_similarity'
            if sim_col in group_sim_df.columns:
                sim_value = group_sim_df[sim_col].iloc[0]
                if pd.notna(sim_value):
                    dim_name = emb_type.replace('_', ' ').title()
                    print(f"{dim_name:20s}: {sim_value:.4f}")
    else:
        print(f"\n群体相似度文件不存在: {group_similarities_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="用户扩量示例")
    parser.add_argument("--mode", choices=["run", "analyze"], default="run",
                       help="运行模式: run=运行扩量, analyze=分析结果")
    parser.add_argument("--embeddings_path", type=str,
                       default="experiments/location_feature/models/comprehensive_user_embeddings.pkl",
                       help="用户向量文件路径")
    parser.add_argument("--results_path", type=str,
                       default="expansion_results/expansion_results.csv",
                       help="结果文件路径（用于分析模式）")
    
    args = parser.parse_args()
    
    if args.mode == "run":
        run_expansion_example()
    elif args.mode == "analyze":
        analyze_results(args.results_path)
