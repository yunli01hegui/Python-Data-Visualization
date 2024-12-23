import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

car_data = pd.read_csv("./../datas/second_cars_info.csv", encoding="gbk")
print(car_data)

# 先筛选出 Boarding_time 列为 "未上牌" 的数据，然后使用 count 方法统计行数
count_result = car_data[car_data.Boarding_time == "未上牌"].count()
# 通过 iloc[0] 按位置获取统计结果中的第一个（也是唯一的一个）值，即符合条件的行数
N = count_result.iloc[0]
Ration = N / car_data.shape[0]
print(Ration)

cars = car_data.loc[car_data.Boarding_time != '未上牌', :]
print(cars)

car = cars.copy()
car.index = range(0, car.shape[0])
car['year'] = car.Boarding_time.str[:4].astype('int')
month = car.Boarding_time.str.findall('年(.*?)月')
month = pd.DataFrame([i[0] for i in month]).astype('int')
car['month'] = month
print(car.head())

car['New_price'] = car['New_price'].str.extract('(\\d+\\.?\\d+)', expand=True)
car['Km'] = car['Km'].str.extract('(\\d+\\.?\\d+)', expand=True)

# 定义一个函数用于去除字符串中的单位并转换为数值
def convert_to_float(x):
    if isinstance(x, str):
        x = x.replace('万', '')
    return float(x)


# 对每一列应用这个函数来转换数据类型
car['New_price'] = car['New_price'].apply(convert_to_float)
car['Sec_price'] = car['Sec_price'].apply(convert_to_float)
car['Km'] = car['Km'].apply(convert_to_float)


# 确保 'car' 是一个已定义的 DataFrame，并且包含 'Brand' 列
# car = pd.read_csv('your_file.csv')  # 示例加载数据的方式
# 设置字体以支持中文显示
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei']
# 修改此处，使用推荐的方式获取品牌销量前十的数据
car_top10 = pd.Series(car["Brand"]).value_counts().sort_values(ascending=False)[:10]
# 使用更简洁的方法创建 brand 和 brand_num 列表
brand = list(car_top10.index) + ["其他"]
brand_num = list(car_top10.values) + [car.shape[0] - sum(car_top10.values)]
# 将品牌和数量组合成字典并转换为 Series 对象
data = dict(zip(brand, brand_num))
df = pd.Series(data)
# 绘制饼图
plt.figure(figsize=(10, 10))
df.plot(kind="pie", title="车辆销售品牌占比", autopct="%.2f%%")
plt.ylabel('')  # 移除默认的 y 标签
plt.show()


car_year = car['year'].value_counts()
car_year = car_year.to_frame().sort_index()
car_year.plot(figsize=(12, 8), kind='bar', title="车龄")
plt.show()


index = car['Brand'].isin(['奥迪'])
some_cars = car.loc[index, :]
some_cars
some_cars.plot.scatter(
    figsize=(12, 8),
    x='Km', y='Sec_price',
    s=20, marker='o',
    alpha=0.9,
    linewidths=0.3,
    edgecolors='k'
)
plt.show()

# --------------------------------------------------------------------------------------
# 读取数据
data = pd.read_csv("./../datas/second_cars_info.csv", encoding="gbk")

# 使用matplotlib绘制不同车型的里程（Km）的折线图（多条折线，每个车型一条，示例中选取部分车型示意，可根据需求调整）
# 选取部分车型进行示例展示，你可以根据实际需求修改这里选取的车型
selected_models = ['奥迪A6L 2006款 2.4 CVT 舒适型', '奥迪A8L 2013款 45 TFSI quattro舒适型', '奥迪Q5 2013款 40TFSI 舒适型']
subset_data = data[data['Name'].isin(selected_models)]

# 绘制折线图
for model in selected_models:
    model_data = subset_data[subset_data['Name'] == model]
    plt.plot(model_data['Km'], label=model)

plt.xlabel('车辆序号（示例）')
plt.ylabel('里程（万公里）')
plt.title('部分奥迪车型里程变化情况')
plt.legend()
plt.show()

# 使用seaborn绘制不同品牌（这里都是奥迪，可拓展为多品牌对比情况）与价格（New_price）的关系小提琴图（示例中可看到价格分布情况）
# 绘制小提琴图，这里只有奥迪品牌，可拓展为多品牌对比
sns.violinplot(x='Brand', y='New_price', data=data)
plt.xlabel('品牌')
plt.ylabel('新车价格（万）')
plt.title('奥迪品牌车型新车价格分布（小提琴图）')
plt.show()


# ---------------------------------------------------------------------
# 雷达图
labels = ["A", "B", "C", "D"]
num_vars = len(labels)

values = [3, 2, 4, 5]  # 示例数据
values += values[:1]  # 闭合曲线

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合曲线

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color="red", alpha=0.25)
ax.plot(angles, values, color="red", linewidth=2)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
plt.show()

# 箱线图
data = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # 示例数据
plt.boxplot(data)
plt.title("价格分布情况")
plt.ylabel("价格")
plt.show()

# 热力图（需要安装seaborn库）
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 示例数据
df = pd.DataFrame(data, columns=['Jan', 'Feb', 'Mar'])
heatmap_data = df.T
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
plt.title("月度销售热度")
plt.show()

# ---------------------------------------------------------------------


