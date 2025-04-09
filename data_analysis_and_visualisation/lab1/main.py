import json
import re

import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import matplotlib.pyplot as plt

BASE_URL = "https://books.toscrape.com/"
books_data = []

for page in range(1, 51):
    url = f"{BASE_URL}catalogue/page-{page}.html"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Ошибка загрузки {url}")
        continue

    soup = bs(response.text, "html.parser")
    books = soup.find_all("article", class_="product_pod")

    for book in books:
        title = book.h3.a["title"]
        price = book.find("p", class_="price_color").text.strip().replace("Â", "")
        availability = book.find("p", class_="instock availability").text.strip()
        rating_class = book.find("p", class_="star-rating")["class"]
        rating = rating_class[1] if len(rating_class) > 1 else "No rating"
        link = BASE_URL + book.h3.a["href"]
        img_src = BASE_URL + book.find("img")["src"].replace("../", "")

        books_data.append({
            "title": title,
            "price": price,
            "availability": availability,
            "rating": rating,
            "link": link,
            "image": img_src
        })
        print(f"Parsed book {title} on page - {page}")

with open("books.json", "w", encoding="utf-8") as json_file:
    json.dump(books_data, json_file, ensure_ascii=False, indent=4)


df = pd.DataFrame(books_data)

df["price"] = df["price"].apply(lambda x: float(re.sub(r'[^\d.]', '', x)))

mean_price = df["price"].mean()
median_price = df["price"].median()
mode_price = df["price"].mode().values[0] 
std_dev_price = df["price"].std(ddof=1)
min_price = df["price"].min()
max_price = df["price"].max()

books_in_stock = df["availability"].str.contains("In stock").sum()
books_out_of_stock = df["availability"].str.contains("Out of stock").sum()

most_common_rating = df["rating"].mode().values[0]

print(f"Mean Price: {mean_price:.2f}")
print(f"Median Price: {median_price:.2f}")
print(f"Mode Price: {mode_price:.2f}")
print(f"Standard Deviation: {std_dev_price:.2f}")
print(f"Min Price: {min_price:.2f}")
print(f"Max Price: {max_price:.2f}")
print(f"Total Books: {len(df)}")
print(f"Books In Stock: {books_in_stock}")
print(f"Books Out of Stock: {books_out_of_stock}")
print(f"Most Common Rating: {most_common_rating}")


plt.figure(figsize=(8, 5))
plt.boxplot(df["price"], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("Box Plot of Book Prices")
plt.xlabel("Price (£)")
plt.savefig("box_plot.png")
plt.close()

rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
df["rating_numeric"] = df["rating"].map(rating_map)

plt.figure(figsize=(8, 5))
df['rating_numeric'].dropna().astype(int).value_counts().sort_index().plot(kind='bar', color='coral', edgecolor='black')
plt.title('Book Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig("rating_plot.png")
plt.close()


plt.figure(figsize=(8, 5))
plt.scatter(df["rating_numeric"], df["price"], color="red", alpha=0.6, edgecolors="black")
plt.title("Scatter Plot of Price vs. Rating")
plt.xlabel("Rating (Stars)")
plt.ylabel("Price (£)")
plt.xticks([1, 2, 3, 4, 5])
plt.grid(True)
plt.savefig("rating_price_plot.png")
plt.close()