#!/bin/ksh
# Demo ETL orchestrator. Illustration only: the analyzer parses this statically
# and never executes it. It loads the two source tables with embedded SQL, runs
# the transform SQL file, then invokes a script and a function that are
# intentionally absent from the project to illustrate missing references.

load_customer() {
    bteq <<EOF
INSERT INTO demo.t_customer (customer_id, customer_name)
VALUES (1, 'Generic Customer');
EOF
}

load_orders() {
    bteq <<EOF
INSERT INTO demo.t_orders (order_id, customer_id, amount)
VALUES (10, 1, 100);
EOF
}

build() {
    bteq < ./transform.sql
}

main() {
    load_customer
    load_orders
    build
    ./missing_lib.sh
    notify_done
}

main
