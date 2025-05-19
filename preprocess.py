# 预处理 parquet 文件
import pyarrow.parquet as pq
import pandas as pd
import os

file_folder = 'your path of dataset'

# 只保留 parquet 文件（过滤文件夹/非parquet文件）
parquet_files = [f for f in os.listdir(file_folder) if f.endswith('.parquet')]

# 使用生成器表达式替代列表推导式节省内存，并指定读取的列
tables = (
    pq.read_table(
        os.path.join(file_folder, pf),
        columns=['purchase_history', 'registration_date', 'login_history']  # 明确指定要读取的列
    )
    for pf in parquet_files
)

# 使用更高效的分块合并方式
df = pd.concat(
    (table.to_pandas() for table in tables),
    ignore_index=True
)

import pandas as pd
from datetime import datetime
import json

from tqdm import tqdm

def preprocess(df, catalog_file):
    # 处理 purchase_history 列
    import json
    from datetime import datetime

    with open(catalog_file, 'r', encoding='utf-8') as f:
        product_catalog = json.load(f)
    # 1. 解析商品目录为DataFrame并建立id到类别和价格的映射
    catalog_df = pd.DataFrame(product_catalog['products'])
    catalog_df.set_index('id', inplace=True)
    # 生成映射字典
    id2cat = catalog_df['category'].to_dict()
    id2price = catalog_df['price'].to_dict()

    # 6. 分类主类别（逻辑不变）
    categories = {
        "电子产品": ["智能手机","笔记本电脑","平板电脑","智能手表","耳机","音响","相机","摄像机","游戏机"],
        "服装": ["上衣","裤子","裙子","内衣","鞋子","帽子","手套","围巾","外套"],
        "食品": ["零食","饮料","调味品","米面","水产","肉类","蛋奶","水果","蔬菜"],
        "家居": ["家具","床上用品","厨具","卫浴用品"],
        "办公": ["文具","办公用品"],
        "运动户外": ["健身器材", "户外装备"],
        "玩具": ["玩具","模型","益智玩具"],
        "母婴": ["婴儿用品","儿童课外读物"],
        "汽车用品": ["车载电子","汽车装饰"]
    }
    # 逆向映射，将数组中的每一类物品映射到对应的类别
    category_mapping = {item: category for category, items in categories.items() for item in items}

    def parse_history(ph_str):
        try:
            data = json.loads(ph_str)
            # 提取核心字段
            items = [category_mapping[id2cat[item['id']]] for item in data['items'] if item['id'] in id2cat]
            return {
                'categories': items,
                'prices': [id2price[item['id']] for item in data['items'] if item['id'] in id2price],
                'payment_method': data.get('payment_method', '未知'),
                'payment_status': data.get('payment_status', '未知'),
                'purchase_date': datetime.strptime(data['purchase_date'], "%Y-%m-%d"),
            }
        except Exception as e:
            print(f"解析异常: {str(e)}")
            return None

    # 应用解析函数并展开JSON字段
    tqdm.pandas(desc="purchase_history 处理进度")
    parsed_data = df['purchase_history'].progress_apply(parse_history)
    
    # 合并解析结果到新DataFrame
    processed_df = pd.json_normalize(parsed_data)
    
    # 增加两列'login_history', 'registration_date'

    return processed_df[['categories', 'payment_method', 'payment_status', 
                       'purchase_date', 'prices']].join(df[['login_history', 'registration_date']]) 

def generate_temporal_mask(df):
    # 从login_history提取最后登录时间（修正注释错误）
    def extract_last_login(lh_str):
        try:
            data = json.loads(lh_str)
            timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in data.get('timestamps', [])]
            return max(timestamps) if timestamps else None
        except:
            return None
    tqdm.pandas(desc="login_history 处理进度")
    df['last_login'] = df['login_history'].progress_apply(extract_last_login)
    
    # 统一时间转换
    time_cols = ['registration_date', 'purchase_date', 'last_login']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # 生成掩码（排除无效时间）
    mask = (
        (df['registration_date'] <= df['purchase_date']) & 
        (df['purchase_date'] <= df['last_login']) &
        df[time_cols].notnull().all(axis=1)
    )
    return mask

df = preprocess(df, 'product_catalog.json')
valid_mask = generate_temporal_mask(df)

# 5. 过滤并保存
processed_df = df[valid_mask].reset_index(drop=True)
processed_df = processed_df[['categories', 'payment_method', 'payment_status', 
                       'purchase_date', 'prices']]
processed_df.to_parquet('processed_df.parquet', index=False)