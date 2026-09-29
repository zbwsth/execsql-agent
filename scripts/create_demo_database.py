"""Create the deterministic SQLite database used by demos and tests."""

from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DEFAULT_OUTPUT = Path("data/demo.db")

CUSTOMERS = [
    (1, "张伟", "北京", "2023-01-15"),
    (2, "王芳", "上海", "2023-02-20"),
    (3, "李娜", "广州", "2023-03-08"),
    (4, "刘洋", "深圳", "2023-04-11"),
    (5, "陈晨", "杭州", "2023-05-19"),
    (6, "杨帆", "成都", "2023-06-23"),
    (7, "赵敏", "武汉", "2023-07-14"),
    (8, "黄杰", "南京", "2023-08-05"),
    (9, "周静", "西安", "2023-09-17"),
    (10, "吴昊", "苏州", "2023-10-09"),
]

PRODUCTS = [
    (1, "机械键盘", "电脑配件", 399.0),
    (2, "无线鼠标", "电脑配件", 159.0),
    (3, "27寸显示器", "显示设备", 1899.0),
    (4, "USB-C 扩展坞", "电脑配件", 499.0),
    (5, "降噪耳机", "音频设备", 1299.0),
    (6, "高清摄像头", "视频设备", 599.0),
    (7, "人体工学椅", "办公家具", 2399.0),
    (8, "升降桌", "办公家具", 3299.0),
]

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    city TEXT NOT NULL,
    signup_date TEXT NOT NULL
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL CHECK (unit_price >= 0)
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'pending', 'cancelled')),
    total_amount REAL NOT NULL CHECK (total_amount >= 0),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price REAL NOT NULL CHECK (unit_price >= 0),
    line_amount REAL NOT NULL CHECK (line_amount >= 0),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
"""


def _build_orders() -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    prices = {product_id: price for product_id, _, _, price in PRODUCTS}
    orders: list[tuple[object, ...]] = []
    items: list[tuple[object, ...]] = []
    item_id = 1
    start_date = date(2025, 1, 1)

    for order_id in range(1, 31):
        customer_id = ((order_id * 3 - 1) % len(CUSTOMERS)) + 1
        product_ids = (
            ((order_id * 2 - 1) % len(PRODUCTS)) + 1,
            ((order_id * 2 + 2) % len(PRODUCTS)) + 1,
        )
        quantities = ((order_id % 3) + 1, ((order_id + 1) % 4) + 1)
        line_amounts: list[float] = []

        for product_id, quantity in zip(product_ids, quantities, strict=True):
            unit_price = prices[product_id]
            line_amount = round(unit_price * quantity, 2)
            line_amounts.append(line_amount)
            items.append(
                (item_id, order_id, product_id, quantity, unit_price, line_amount)
            )
            item_id += 1

        if order_id % 7 == 0:
            status = "cancelled"
        elif order_id % 5 == 0:
            status = "pending"
        else:
            status = "completed"
        order_date = (start_date + timedelta(days=order_id * 3)).isoformat()
        orders.append(
            (order_id, customer_id, order_date, status, round(sum(line_amounts), 2))
        )

    return orders, items


def create_demo_database(output_path: str | Path = DEFAULT_OUTPUT) -> Path:
    """Create a fresh deterministic demo database and atomically replace the target."""

    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    if temporary.exists():
        temporary.unlink()

    orders, order_items = _build_orders()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(SCHEMA_SQL)
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?)", CUSTOMERS
        )
        connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", PRODUCTS)
        connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)
        connection.executemany(
            "INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)", order_items
        )
        connection.commit()
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    os.replace(temporary, target)
    return target


def main() -> None:
    """Parse CLI arguments and create the demo database."""

    parser = argparse.ArgumentParser(description="创建确定性的 ExecSQL-Agent Demo 数据库。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="SQLite 输出路径")
    args = parser.parse_args()
    created_path = create_demo_database(args.output)
    print(f"Demo database created: {created_path}")


if __name__ == "__main__":
    main()
