import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# python train_label_discriminator.py --embeddings_path experiments\node2vec\models\user_embeddings_node2vec.pkl --labels_path data\label.csv


def load_data(embeddings_path, labels_path):
    """Loads user embeddings and labels, then merges them."""
    try:
        with open(embeddings_path, 'rb') as f:
            user_embeddings_dict = pickle.load(f)
        print(f"成功从 {embeddings_path} 加载嵌入向量")
        print(f"包含嵌入向量的用户数量: {len(user_embeddings_dict)}")
    except FileNotFoundError:
        print(f"错误: 在 {embeddings_path} 未找到嵌入向量文件")
        return None, None
    except Exception as e:
        print(f"加载嵌入向量时出错: {e}")
        return None, None

    try:
        labels_df = pd.read_csv(labels_path, encoding='utf-8', sep='\t')
        print(f"成功从 {labels_path} 加载标签")
        print(f"包含标签的用户数量: {len(labels_df)}")
        if 'user_id' not in labels_df.columns or 'label' not in labels_df.columns:
            print("错误: 标签文件必须包含 'user_id' 和 'label' 列。")
            return None, None
    except FileNotFoundError:
        print(f"错误: 在 {labels_path} 未找到标签文件")
        return None, None
    except Exception as e:
        print(f"加载标签时出错: {e}")
        return None, None

    # 将 labels_df 中的 user_id 转换为字符串以匹配 user_embeddings_dict 中的键
    labels_df['user_id'] = labels_df['user_id'].astype(str)

    X_list = []
    y_list = []
    
    missing_embeddings_count = 0
    processed_users = set()

    for index, row in labels_df.iterrows():
        user_id = row['user_id']
        label = row['label']
        
        if user_id in processed_users:
            print(f"警告: 用户 {user_id} 在标签文件中重复出现，将只使用第一次出现的标签。")
            continue
        processed_users.add(user_id)

        if user_id in user_embeddings_dict:
            X_list.append(user_embeddings_dict[user_id])
            y_list.append(label)
        else:
            missing_embeddings_count += 1
    
    if missing_embeddings_count > 0:
        print(f"警告: 来自标签文件的 {missing_embeddings_count} 个用户在嵌入向量文件中未找到，已被跳过。")

    if not X_list:
        print("错误: 在嵌入向量和标签数据之间未找到共同用户，或合并后数据集为空。")
        return None, None
        
    X = np.array(X_list)
    y = np.array(y_list) # 初始为原始标签
    
    # 如果标签不是数字类型，则进行编码
    # 首先检查y是否为空，以及它的dtype
    if y.size > 0 and not np.issubdtype(y.dtype, np.number):
        print("标签不是数字类型。正在应用 LabelEncoder。")
        le = LabelEncoder()
        y_transformed = le.fit_transform(y)
        print(f"标签映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        y_numeric = y_transformed
    elif y.size > 0 and np.issubdtype(y.dtype, np.number):
        print("标签是数字类型。")
        y_numeric = y.astype(int) # 确保是整数类型以用于分箱等操作
    else: # y.size == 0
        print("错误：没有可用的标签数据。")
        return None, None


    print(f"准备好的数据包含 {X.shape[0]} 个样本和 {X.shape[1]} 个特征。")
    return X, y_numeric

def train_and_evaluate_discriminator(X, y, test_size=0.2, random_state=42):
    """Trains a discriminator and evaluates its accuracy."""
    if X is None or y is None or len(X) == 0:
        print("无法训练和评估：输入数据无效或为空。")
        return

    if len(X) != len(y):
        print(f"错误: X ({len(X)}) 和 y ({len(y)}) 的样本数量不匹配。")
        return
        
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        print("错误: 分类至少需要两个类别。当前只找到一个类别。")
        if len(y) > 0:
            print(f"唯一的类别值: {unique_labels[0]}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"训练数据形状: {X_train.shape}, 测试数据形状: {X_test.shape}")
    
    # 确保标签是整数类型以便使用np.bincount
    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)

    print(f"训练集标签分布: {np.bincount(y_train_int) if len(y_train_int) > 0 else '空训练集标签'}")
    print(f"测试集标签分布: {np.bincount(y_test_int) if len(y_test_int) > 0 else '空测试集标签'}")

    model = LogisticRegression(random_state=random_state, max_iter=1000, solver='saga', verbose=10) 
    
    print(f"使用模型: {type(model).__name__} (优化器: saga)")
    print("开始训练判别器...")
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        print(f"模型训练期间发生错误: {e}")
        print("这可能是由于分割后所有样本都属于一个类别，或数据中存在NaN/inf值等问题造成的。")
        return

    print("模型训练完成。")
    print("开始在测试集上进行预测...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"预测完成。")
    print(f"\n判别器测试准确率: {accuracy:.4f}")

    return accuracy

def main():
    parser = argparse.ArgumentParser(description="在用户嵌入向量和标签上训练一个判别器。")
    parser.add_argument(
        "--embeddings_path", 
        type=str, 
        required=True, 
        help="包含用户嵌入向量的 .pkl 文件路径 (格式: user_id -> embedding 的字典)。"
    )
    parser.add_argument(
        "--labels_path", 
        type=str, 
        required=True, 
        help="包含用户标签的 .csv 文件路径 (列: user_id, label)。"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="数据集中用于测试集的比例 (默认: 0.2)。"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="用于可复现性的随机种子 (默认: 42)。"
    )
    
    args = parser.parse_args()
    
    print("开始判别器训练流程...")
    X, y = load_data(args.embeddings_path, args.labels_path)
    
    if X is not None and y is not None and X.shape[0] > 0 :
        train_and_evaluate_discriminator(X, y, test_size=args.test_size, random_state=args.random_state)
    else:
        print("由于数据加载问题或合并后数据集为空，无法继续训练。")
        
    print("流程结束。")

if __name__ == "__main__":
    main() 