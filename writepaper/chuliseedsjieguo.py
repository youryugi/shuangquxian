import pandas as pd

# 读取CSV（跳过第2行注释行）
df = pd.read_csv(
    r"C:\Users\79152\Desktop\github\shuangquxian\ml\0409\0519-1_0519_2328\test_metrics_summary.csv",
    
    skiprows=[1]  # 跳过第2行（中英文注释行）
)

# 指标列
metric_cols = [
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "global_iou",
]

# 按 lambda_size 分组，计算各指标均值
result = df.groupby("lambda_size")[metric_cols].mean().reset_index()
result = result.sort_values("lambda_size").reset_index(drop=True)

# 打印结果
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

print("=" * 80)
print("各 lambda 值在不同种子下的指标均值（10个种子）")
print("=" * 80)
print(result.to_string(index=False))

# 同时保存为CSV
result.to_csv("lambda_metrics_mean.csv", index=False, float_format="%.6f")
print("\n结果已保存至 lambda_metrics_mean.csv")