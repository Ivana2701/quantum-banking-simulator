-- demo addresses
INSERT INTO addresses
    (street, city, state, country, postal_code)
VALUES
    ('123 Main St', 'Sofia', '', 'Bulgaria', '1000'),
    ('456 Elm St', 'Plovdiv', '', 'Bulgaria', '4000');

-- demo phone numbers
INSERT INTO phone_numbers
    (number, type)
VALUES
    ('+359888123456', 'mobile'),
    ('+359298765432', 'work');

-- demo geolocations
INSERT INTO geolocations
    (latitude, longitude, description)
VALUES
    (42.6977, 23.3219, 'Sofia HQ'),
    (42.1354, 24.7453, 'Plovdiv Branch');

-- demo IP addresses
INSERT INTO ip_addresses
    (ip_address)
VALUES
    ('193.144.10.1'),
    ('193.144.10.2');

-- demo devices
INSERT INTO devices
    (device_name, fingerprint)
VALUES
    ('Alice Laptop', 'fp_alice_001'),
    ('Bob Phone', 'fp_bob_002');

-- demo accounts
INSERT INTO accounts
    (
    encrypted_balance, username, full_name, password_hash,
    account_type, role_id
    )
VALUES
    (
        decode('00AABBCC','hex'), 'alice', 'Alice Ivanova',
        '$2b$12$abcdefghijklmnopqrstuv', 'customer', 1
  ),
    (
        decode('00112233','hex'), 'bob', 'Bob Petrov',
        '$2b$12$qrstuvwxyzabcdefghi', 'employee', 2
  ),
    (
        decode('FFEE0011','hex'), 'carol', 'Carol Dimitrova',
        '$2b$12$1234567890abcdefghi', 'admin', 3
  );

-- map accounts to their contact info
INSERT INTO account_addresses
    (account_id, address_id)
VALUES
    (1, 1),
    (2, 2);

INSERT INTO account_phones
    (account_id, phone_id)
VALUES
    (1, 1),
    (2, 2);

INSERT INTO account_geos
    (account_id, geo_id)
VALUES
    (1, 1),
    (2, 2);

INSERT INTO account_ips
    (account_id, ip_id)
VALUES
    (1, 1),
    (2, 2);

INSERT INTO account_devices
    (account_id, device_id)
VALUES
    (1, 1),
    (2, 2);

-- demo encrypted transactions and their keys
INSERT INTO transactions_encrypted
    (
    transaction_id, account_id, amount, encrypted_data, recorded_at,
    from_account_id, to_account_id, is_fraud
    )
VALUES
    (
        1, 1, 150.75,
        decode('DEADBEEF','hex'), now(),
        1, 2, FALSE
  ),
    (
        2, 2, 200.00,
        decode('CAFEBABE','hex'), now(),
        2, 1, FALSE
  );

INSERT INTO transaction_keys
    (transaction_id, bb84_key, created_at)
VALUES
    (1, decode('BAADF00D','hex'), now()),
    (2, decode('FEEDFACE','hex'), now());

-- demo immutable transaction history (mirrors encrypted set)
INSERT INTO transactions
    (
    transaction_id, account_id, from_account_id, to_account_id,
    encrypted_amount, is_fraud, created_at
    )
VALUES
    (
        1, 1, 1, 2,
        decode('DEADBEEF','hex'), FALSE, now()
  ),
    (
        2, 2, 2, 1,
        decode('CAFEBABE','hex'), FALSE, now()
  );

-- demo QML stream
INSERT INTO transactions_qml
    (
    amount, transaction_time, geo_location, ip_address, is_fraud
    )
VALUES
    (150.75, now(), 'Sofia HQ', '193.144.10.1', FALSE),
    (200.00, now(), 'Plovdiv Branch', '193.144.10.2', FALSE);

-- demo classification metrics
INSERT INTO classification_metrics
    (
    algorithm_type_id, accuracy, "precision", recall, f1_score,
    confusion_matrix, roc_curve
    )
VALUES
    (
        1, 0.95, 0.92, 0.90, 0.91,
        '{"tp":95,"fp":5,"tn":90,"fn":10}'
::jsonb,
    '{"points":[[0,0],[1,1]]}'::jsonb
  ),
(
    2, 0.93, 0.90, 0.88, 0.89,
    '{"tp":90,"fp":10,"tn":85,"fn":15}'::jsonb,
    '{"points":[[0,0],[1,1]]}'::jsonb
  );
