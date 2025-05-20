import pyarrow.parquet as pq
import pandas as pd
import os
import itertools
import time
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler
from itertools import cycle
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 设置字体，确保中文显示正常
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12 
plt.rcParams['figure.figsize'] = (12, 8)  # 默认图表大小
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

# 自定义配色
custom_colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", 
                "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

# 自定义渐变配色
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', 
                                              ['#E8F8F5', '#1ABC9C', '#117864'], 
                                              N=256)

# 可视化关联规则
def visualize_rules(rules, title, filename, max_rules=20):
    """
    可视化关联规则 - 改进版
    """
    print(f"可视化关联规则: {title}...")
    
    if rules.empty:
        print("没有规则可视化")
        return
    
    # 限制规则数量
    rules_to_plot = rules.head(max_rules)
    
    # 1. 3D散点图：支持度-置信度-提升度
    plt.figure(figsize=(14, 10))
    ax = plt.axes(projection='3d')
    
    # 计算气泡大小和颜色
    sizes = rules_to_plot['lift'].values * 300 # 提升度越高，气泡越大
    colors = rules_to_plot['confidence'].values # 置信度越高，颜色越深
    
    # 创建3D散点图
    scatter = ax.scatter(
        rules_to_plot['support'], 
        rules_to_plot['confidence'], 
        rules_to_plot['lift'],
        s=sizes, 
        c=colors, 
        alpha=0.7, 
        cmap=custom_cmap,
        edgecolors='w'
    )
    
    # 添加标签 - 只为前10个规则添加标签，避免拥挤
    for i, row in rules_to_plot.head(10).iterrows():
        ax.text(row['support'], row['confidence'], row['lift'], 
               f"{row['antecedents']} -> {row['consequents']}", 
               fontsize=12,
               alpha=0.8,
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
    ax.set_xlabel('支持度', fontsize=14, labelpad=10)
    ax.set_ylabel('置信度', fontsize=14, labelpad=10)
    ax.set_zlabel('提升度', fontsize=14, labelpad=10)
    
    # 设置刻度标签大小
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, label='置信度', shrink=0.5, pad=0.1)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel('置信度', fontsize=14)
    
    plt.title(f'{title} - 3D规则可视化', fontsize=16, pad=20)
    plt.tight_layout()
    
    # 保存3D图像
    plt.savefig(f'{filename}_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 气泡图：支持度-置信度 (REMOVED)
    plt.figure(figsize=(14, 10))
    # 使用colormap为提升度
    norm = plt.Normalize(rules_to_plot['lift'].min(), rules_to_plot['lift'].max())
    # 改造成迭代格式，防止颜色数不够
    colors_gen = (plt.cm.viridis(norm(val)) for val in rules_to_plot['lift'])

    # 创建散点图
    for i, row in rules_to_plot.iterrows():
        temp_color = next(colors_gen)
        plt.scatter(
            row['support'], 
            row['confidence'], 
            # 根据最大值设置气泡大小
            s=row['lift'] * 320 * 3 / rules_to_plot['lift'].max(), 
            color=temp_color,
            alpha=0.7,
            edgecolors='w'
        )
        
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='提升度')
    
    plt.xlabel('支持度')
    plt.ylabel('置信度')
    plt.tick_params(axis='x', labelrotation=0)
    plt.tick_params(axis='y', labelrotation=0)
    # 设置标题
    plt.title(f'{title} - 前{len(rules_to_plot)}条规则的指标关系', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存气泡图
    plt.savefig(f'{filename}_bubble.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 热力图展示前10条规则
    top_rules_heatmap = rules_to_plot.head(10).copy()
    top_rules_table = rules_to_plot.head(20).copy()

    plt.figure(figsize=(10, 10))
    
    # 准备热力图数据
    rule_labels = [f"{row['antecedents']} -> {row['consequents']}" for _, row in top_rules_heatmap.iterrows()]
    # leverage：杠杆值，用于衡量项集 A 和 B 的共现频率与独立出现时的期望频率之间的差异，>0表示正相关，<0表示负相关
    # conviction: 衡量规则 A→B 的不可靠程度，值越大规则越可靠
    metrics = ['support', 'confidence', 'lift', 'leverage', 'conviction']

    def calculate_metrics(top_rules_heatmap):
        # 确保所有指标都存在
        for metric in metrics:
            if metric not in top_rules_heatmap.columns:
                if metric == 'leverage':
                    if 'antecedent support' in top_rules_heatmap.columns and 'consequent support' in top_rules_heatmap.columns:
                        top_rules_heatmap[metric] = top_rules_heatmap.apply(lambda x: 
                                                        float(x['support']) - 
                                                        float(x['antecedent support']) * 
                                                        float(x['consequent support']), axis=1)
                    else:
                        top_rules_heatmap[metric] = 0
                elif metric == 'conviction':
                    if 'consequent support' in top_rules_heatmap.columns:
                        top_rules_heatmap[metric] = top_rules_heatmap.apply(lambda x: 
                                                        (1 - float(x['consequent support'])) / 
                                                        (1 - float(x['confidence'])) 
                                                        if float(x['confidence']) < 1 else float('inf'), axis=1)
                        top_rules_heatmap[metric] = top_rules_heatmap[metric].replace(float('inf'), 
                            top_rules_heatmap[metric][top_rules_heatmap[metric] != float('inf')].max() * 2 
                            if any(top_rules_heatmap[metric] != float('inf')) else 2)
                    else:
                        top_rules_heatmap[metric] = 0
        return top_rules_heatmap
    
    top_rules_heatmap = calculate_metrics(top_rules_heatmap)
    top_rules_table = calculate_metrics(top_rules_table)
    
    # 提取指标值
    heatmap_data = top_rules_heatmap[metrics].values
    
    # 获取行列数
    num_rows = len(metrics)
    num_cols = len(rule_labels)
    # 设置单个网格的大小
    cell_size = 1.5  # 可根据需要调整，如 1.0, 1.2, 1.5 等
    # 动态计算画布大小
    figsize = (num_cols * cell_size, num_rows * cell_size)
    plt.figure(figsize=figsize)
    # 绘制热力图
    ax = sns.heatmap(
        heatmap_data.T,
        annot=True,
        fmt='.3f',
        yticklabels=metrics,
        xticklabels=range(1, len(rule_labels) + 1),
        cmap='YlGnBu',
        annot_kws={'size': 14},
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    plt.tick_params(axis='x', labelrotation=0)
    plt.tick_params(axis='y', labelrotation=0)
    # 设置标题
    plt.title(f'{title} - 前{len(rules_to_plot)}条规则的多指标比较', fontsize=16, pad=20)
    # 自动调整布局
    plt.tight_layout()
    plt.savefig(f'{filename}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 规则网络图
    try:
        plt.figure(figsize=(16, 12))
        G = nx.DiGraph()
        
        # 添加节点
        all_items = set()
        for _, row in rules_to_plot.iterrows():
            antecedents = row['antecedents'].split(',')
            consequents = row['consequents'].split(',')
            all_items.update(consequents)
        
        for item in all_items:
            G.add_node(item)
        
        # 添加边
        for _, row in rules_to_plot.iterrows():
            antecedents = row['antecedents'].split(',')
            consequents = row['consequents'].split(',')
            
            for a in antecedents:
                for c in consequents:
                    if G.has_edge(a, c):
                        G[a][c]['weight'] += row['lift']
                        G[a][c]['count'] += 1
                    else:
                        G.add_edge(a, c, weight=row['lift'], count=1)
        
        # 计算节点大小
        node_size = [G.degree(node) * 600 + 1000 for node in G.nodes()]
        
        # 设置布局
        pos = nx.spring_layout(G, k=1.2, seed=42)
        
        # 设置边的权重和颜色
        edges = G.edges(data=True)
        weights = [data['weight'] * 0.8 for _, _, data in edges]
        counts = [data['count'] for _, _, data in edges]
        
        # 绘制节点
        node_colors = list(itertools.islice(itertools.cycle(custom_colors), len(G.nodes())))
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, alpha=0.8)
        
        # 绘制边
        edge_colors = plt.cm.Reds([c/max(counts) for c in counts])
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color=edge_colors, 
                              connectionstyle='arc3,rad=0.2', arrowsize=20, arrowstyle='->')
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=16, font_family="SimHei", font_weight='bold')
        
        plt.title(f'{title} - 规则关系网络图', fontsize=18, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{filename}_network.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"绘制网络图时出错: {e}")
    
    # 5. 详细规则表格
    plt.figure(figsize=(18, len(top_rules_table) * 0.8 + 2))
    ax = plt.gca()
    ax.axis('off')
    
    # 设置列宽
    col_widths = [0.3, 0.14, 0.14, 0.14, 0.14, 0.14]
    
    # 创建表格
    table_data = []
    table_data.append(['规则', '结果', '支持度', '置信度', '提升度', '说明'])
    
    for i, row in top_rules_table.iterrows():
        # 解释规则的强度
        if row['lift'] > 5:
            explanation = "非常强的关联"
        elif row['lift'] > 3:
            explanation = "强关联"
        elif row['lift'] > 1.5:
            explanation = "中等关联"
        elif row['lift'] > 1:
            explanation = "弱关联"
        else:
            explanation = "无关联"
        
        table_data.append([
            row['antecedents'], 
            row['consequents'], 
            f"{row['support']:.3f}", 
            f"{row['confidence']:.3f}", 
            f"{row['lift']:.3f}",
            explanation
        ])
    
    table = ax.table(cellText=table_data, colWidths=col_widths, loc='center')
    
    # 设置表格属性
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.8)
    
    # 设置表头样式
    for i, cell in enumerate(table._cells[(0, i)] for i in range(len(table_data[0]))):
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold', fontsize=14)
    
    # 设置表格颜色
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if j == 4:  # 提升度列添加颜色
                try:
                    lift_value = float(table_data[i][j])
                    if lift_value > 5:
                        table._cells[(i, j)].set_facecolor('#92D050')
                    elif lift_value > 3:
                        table._cells[(i, j)].set_facecolor('#C6E0B4')
                    elif lift_value > 1.5:
                        table._cells[(i, j)].set_facecolor('#FFEB9C')
                    elif lift_value > 1:
                        table._cells[(i, j)].set_facecolor('#FFC7CE')
                    else:
                        table._cells[(i, j)].set_facecolor('#FF0000')
                except ValueError:
                    pass
    
    ax.add_table(table)
    
    plt.title(f'{title} - 前{len(top_rules_table)}条规则详情', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f'{filename}_table.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. 退款模式分析
def analyze_refund_patterns(processed_df):
    """ 分析与退款状态相关的商品组合模式 """
    # 筛选退款订单（包含两种状态）
    refund_df = processed_df[processed_df['payment_status'].isin(['已退款', '部分退款'])]
    
    if len(refund_df) == 0:
        print("没有可分析的退款记录")
        return None
    
    # 构建事务数据：每个订单的商品类别 + 退款状态
    transactions = []
    for _, row in refund_df.iterrows():
        items = list(row['categories']) 
        items.append(row['payment_status'])  # 将退款状态作为事务项
        transactions.append(items)
    
    # 编码转换
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_fp = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 编码转换
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_fp = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 挖掘频繁项集（支持度≥0.005）
    frequent_itemsets = fpgrowth(df_fp, min_support=0.005, use_colnames=True)

    return frequent_itemsets

def refund_visulize(df, filename_prefix='refund_analysis'):
    """
    分析与退款状态相关的商品类别组合
    """
    print("分析退款模式...")
    
    # 筛选退款订单
    refund_df = df[df['payment_status'].isin(['已退款', '部分退款'])]
    refund_df = refund_df.explode(['categories', 'prices'])
    df = df.explode(['categories', 'prices'])
    
    if refund_df.empty:
        print("没有找到退款记录")
        return
    
    # 计算各类别退款率
    total_by_category = df.groupby('categories').size()
    refund_by_category = refund_df.groupby('categories').size()
    refund_rate = (refund_by_category / total_by_category * 100).fillna(0).sort_values(ascending=False)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x=refund_rate.index, y=refund_rate.values)
    plt.title('各商品类别退款率', fontsize=16, fontweight='bold')
    plt.xlabel('商品类别', fontsize=14)
    plt.ylabel('退款率 (%)', fontsize=14)
    plt.tick_params(axis='x', labelrotation=0)
    plt.tick_params(axis='y', labelrotation=0)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 热力图展示退款商品的价格分布（需要归一化）
    refund_price_dist = pd.cut(refund_df['prices'], bins=[0, 100, 500, 1000, 5000, 10000, float('inf')], 
                               labels=['<100', '100-500', '500-1000', '1000-5000', '5000-10000', '>10000'])
    price_category_cross = pd.crosstab(refund_price_dist, refund_df['categories'])
    
    # 热力图值归一化
    # 按行归一化，使每行和为1
    price_category_cross = price_category_cross.div(price_category_cross.sum(axis=1), axis=0)

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        price_category_cross,
        annot=True, 
        fmt='.2f', 
        cmap='YlGnBu', 
        cbar_kws={'shrink': 0.9}, 
        annot_kws={'size': 14},
        linewidths=0.5,
        linecolor='white',
        square=True
    )
    plt.title('退款商品价格与类别分布', fontsize=16, fontweight='bold')
    plt.xlabel('商品类别', fontsize=14)
    plt.ylabel('价格区间', fontsize=14)
    plt.tick_params(axis='x', labelrotation=0)
    plt.tick_params(axis='y', labelrotation=0)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_price_dist.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_refund_patterns(processed_df):
    """ 分析与退款状态相关的商品组合模式 """
    processed_df = processed_df.explode(['categories', 'prices'])
    
    # 筛选退款订单（包含两种状态）
    refund_df = processed_df[processed_df['payment_status'].isin(['已退款', '部分退款'])]
    
    if len(refund_df) == 0:
        print("没有可分析的退款记录")
        return None
    
    # 构建事务数据：每个订单的商品类别 + 退款状态
    combined_trans = processed_df.apply(
        lambda row: [row['categories'], row['payment_status']],
        axis=1
    )
    
    # 编码转换
    te = TransactionEncoder()
    te_ary = te.fit(combined_trans).transform(combined_trans)
    df_fp = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 挖掘频繁项集（支持度≥0.005）
    frequent_itemsets = fpgrowth(df_fp, min_support=0.005, use_colnames=True)    
    
    return frequent_itemsets

if os.path.exists('processed_df.parquet'):
    processed_df = pq.read_table('processed_df.parquet').to_pandas()
else:
    if os.path.exists('processed_df.csv'):
        processed_df = pd.read_csv('processed_df.csv')

refund_order_df = refund_visulize(processed_df, 'output/refund_patterns')

# 生成关联规则（置信度≥0.4）
refund_frequent_itemsets = analyze_refund_patterns(processed_df) 
patterns_df = pd.DataFrame(refund_frequent_itemsets, columns=["support", "pattern"])
patterns_df.to_csv("output/refund_patterns.csv", index=False)
refund_rules= association_rules(refund_frequent_itemsets, metric="confidence", min_threshold=0.4)
for threshold in [0.4, 0.3, 0.2, 0.1, 0.005]:
    refund_rules = association_rules(refund_frequent_itemsets, metric="confidence", min_threshold=threshold)
    print("尝试置信度阈值:", threshold)
    if not refund_rules.empty:
        refund_rules = refund_rules[
            refund_rules['consequents'].apply(lambda x: x.issubset({'已退款', '部分退款'})) 
            & refund_rules['antecedents'].apply(lambda x: x.isdisjoint({'已退款', '部分退款'}))
        ]
        refund_rules['antecedents'] = refund_rules['antecedents'].apply(lambda x: ', '.join(map(str, x)))
        refund_rules['consequents'] = refund_rules['consequents'].apply(lambda x: ', '.join(map(str, x)))
        print(f"置信度阈值{threshold}成功挖掘到规则!")
        break

# 生成前件为商品组合、后件为退款状态的规则可视化
plt.figure(figsize=(12, 6))
top_rules = refund_rules.nlargest(10, 'lift')

# 自定义标签：将前件转换为商品组合描述
labels = top_rules['antecedents'].apply(
    lambda x: "\n".join([item.strip() for item in str(x).split(',') 
                        if item not in ['已退款', '部分退款']])
)

sns.barplot(
    x='lift', 
    y=labels, 
    hue=top_rules['consequents'].astype(str),  # 按退款状态分类
    data=top_rules, 
    palette='viridis'
)

plt.title('Top 10 商品组合与退款状态关联规则（支持度≥0.005，置信度≥0.4）')
plt.xlabel('提升度（Lift）', fontsize=12)
plt.ylabel('商品组合', fontsize=12)
plt.legend(title='退款状态', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('output/refund_patterns.png', dpi=300)
plt.close()

refund_rules.sort_values(by=['support', 'confidence'], ascending=False)

if not refund_rules.empty:
    # 保存规则到CSV
    refund_rules.to_csv('output/refund_rules.csv', index=False)
    # 可视化规则
    visualize_rules(refund_rules, '退款商品关联规则', 'output/refund_rules')
