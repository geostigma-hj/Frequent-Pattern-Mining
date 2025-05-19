## 项目代码库介绍

### 1. 文件介绍

- `category_analyse.ipynb`：商品类别关联规则挖掘程序
- `category_paymethod.ipynb`：支付方式与商品类别的关联分析程序
- `season_sequential.ipynb`： 时间序列模式挖掘脚本
- `refund.ipynb`：退款模式分析
- `preprocess.py`：数据预处理脚本
- `sequential_output`：时序模式挖掘结果存储文件夹
- `output`：除时序模式外其余程序输出结果存储文件夹
- 剩余的四个 python 文件是其对应 notebook 文件的集成版，方便直接运行，两者内容基本一致

> 除此上述文件外你还需要在当前目录下添加 product_catalog.json 文件才能正常运行代码。

### 2. 代码使用说明

① **数据预处理**

```bash
# 运行之前修改程序中的数据集路径为你自己的路径
python preprocess.py
```

预处理完成后会在当前目录下生成一个`processed_df.parquet`文件，该文件是后续模式挖掘的基础。

② **频繁模式挖掘**

```bash
# 商品类别关联规则挖掘
python category_analyse.py

# 支付方式与商品类别的关联分析
python category_paymethod.py

# 时间序列模式挖掘
python season_sequential.py

# 退款模式分析
python refund.py
```

你也可以运行对应的 notebook，里面有我已经运行好的记录。最终结果会存到 output 和 sequential_output 两文件夹内，具体图片代表含义请自行参照源代码。

