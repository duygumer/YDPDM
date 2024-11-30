import matplotlib.pyplot as plt
import pandas as pd

# CSV dosyalarını okuma
item_categories = pd.read_csv("/Users/duygumert/PycharmProjects/pythonProject3/Predict_future_sales/item_categories.csv")
items = pd.read_csv("/Users/duygumert/PycharmProjects/pythonProject3/Predict_future_sales/items.csv")
sales_train = pd.read_csv("/Users/duygumert/PycharmProjects/pythonProject3/Predict_future_sales/sales_train.csv")
sample_submission = pd.read_csv("/Users/duygumert/PycharmProjects/pythonProject3/Predict_future_sales/sample_submission.csv")
shops = pd.read_csv("/Users/duygumert/PycharmProjects/pythonProject3/Predict_future_sales/shops.csv")
test = pd.read_csv("/Users/duygumert/PycharmProjects/pythonProject3/Predict_future_sales/test.csv")

# İlk 10 satırı görüntüleme
print("Item Categories:")
print(item_categories.head(10))
print("\nItems:")
print(items.head(10))
print("\nSales Train:")
print(sales_train.head(10))
print("\nSample Submission:")
print(sample_submission.head(10))
print("\nShops:")
print(shops.head(10))
print("\nTest:")
print(test.head(10))

# Veri hakkında bilgi
print("\nSales Train Info:")
print(sales_train.info())

# İstatistiksel özet
print("\nSales Train Describe:")
print(sales_train.describe())

# Belirli bir sütunun değerlerini sayma
print("\nValue Counts for 'item_id':")
print(sales_train['item_id'].value_counts())

# Histogram çizimi
sales_train.hist(bins=50, figsize=(15, 10))
plt.suptitle("Sales Train Data Histograms")
plt.show()



