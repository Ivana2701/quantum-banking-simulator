-- 10) A convenient combined ‐ view for “account_profiles”
CREATE VIEW account_profiles
AS
    SELECT
        a.account_id,
        a.username,
        r.role_name,
        c.customer_id,
        c.full_name    AS customer_name,
        c.account_balance,
        e.employee_id,
        e.full_name    AS employee_name,
        e.position,
        a.created_at
    FROM accounts a
        LEFT JOIN roles     r ON a.role_id    = r.role_id
        LEFT JOIN customers c ON c.account_id  = a.account_id
        LEFT JOIN employees e ON e.account_id  = a.account_id;