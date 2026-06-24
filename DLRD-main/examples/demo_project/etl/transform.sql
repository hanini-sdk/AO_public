-- Demo transform: build the focal table from two source tables, then fan out to
-- two report tables. Illustration only; the analyzer parses this statically.

INSERT INTO demo.t_customer_orders (customer_id, customer_name, total_amount)
SELECT c.customer_id, c.customer_name, SUM(o.amount)
FROM demo.t_customer c
JOIN demo.t_orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name;

INSERT INTO demo.t_report_revenue (customer_id, total_amount)
SELECT customer_id, total_amount FROM demo.t_customer_orders;

INSERT INTO demo.t_report_names (customer_name, total_amount)
SELECT customer_name, total_amount FROM demo.t_customer_orders;
