-- remove view
DROP VIEW IF EXISTS account_profiles
CASCADE;

-- remove custom type
DROP TYPE IF EXISTS account_type_enum
CASCADE;

-- drop QML & metrics
DROP TABLE IF EXISTS classification_metrics
CASCADE;
DROP TABLE IF EXISTS transaction_keys
CASCADE;
DROP TABLE IF EXISTS transactions_encrypted
CASCADE;
DROP TABLE IF EXISTS transactions_qml
CASCADE;

-- drop core transactions
DROP TABLE IF EXISTS transactions
CASCADE;

-- drop mapping tables
DROP TABLE IF EXISTS account_devices
CASCADE;
DROP TABLE IF EXISTS account_ips
CASCADE;
DROP TABLE IF EXISTS account_geos
CASCADE;
DROP TABLE IF EXISTS account_phones
CASCADE;
DROP TABLE IF EXISTS account_addresses
CASCADE;

-- drop contact-detail tables
DROP TABLE IF EXISTS devices
CASCADE;
DROP TABLE IF EXISTS ip_addresses
CASCADE;
DROP TABLE IF EXISTS geolocations
CASCADE;
DROP TABLE IF EXISTS phone_numbers
CASCADE;
DROP TABLE IF EXISTS addresses
CASCADE;

-- drop accounts, algorithms, roles
DROP TABLE IF EXISTS accounts
CASCADE;
DROP TABLE IF EXISTS algorithms
CASCADE;
DROP TABLE IF EXISTS roles
CASCADE;

CREATE TABLE roles
(
  role_id SERIAL PRIMARY KEY,
  role_name TEXT NOT NULL
);

INSERT INTO roles
  (role_name)
VALUES
  ('customer'),
  ('employee'),
  ('admin');

CREATE TYPE account_type_enum AS ENUM
('customer','employee','admin');
CREATE TABLE accounts
(
  account_id SERIAL PRIMARY KEY,
  encrypted_balance BYTEA NOT NULL,
  username TEXT NOT NULL UNIQUE,
  full_name TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  account_type account_type_enum NOT NULL,
  role_id INT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  CONSTRAINT fk_role
    FOREIGN KEY (role_id)
    REFERENCES roles(role_id)
);

-- Contact detail tables
CREATE TABLE addresses
(
  address_id SERIAL PRIMARY KEY,
  street TEXT,
  city TEXT,
  state TEXT,
  country TEXT,
  postal_code TEXT
);

CREATE TABLE phone_numbers
(
  phone_id SERIAL PRIMARY KEY,
  NUMBER TEXT NOT NULL,
  TYPE TEXT NOT NULL
  -- e.g. 'mobile','home','work'
);

CREATE TABLE geolocations
(
  geo_id SERIAL PRIMARY KEY,
  latitude DOUBLE PRECISION NOT NULL,
  longitude DOUBLE PRECISION NOT NULL,
  description TEXT
);

CREATE TABLE ip_addresses
(
  ip_id SERIAL PRIMARY KEY,
  ip_address INET NOT NULL,
  first_seen TIMESTAMP NOT NULL DEFAULT now(),
  last_seen TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE devices
(
  device_id SERIAL PRIMARY KEY,
  device_name TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  last_login TIMESTAMP NOT NULL DEFAULT now()
);

-- Mapping tables (many-to-many)
CREATE TABLE account_addresses
(
  account_id INT NOT NULL,
  address_id INT NOT NULL,
  PRIMARY KEY (account_id, address_id),
  FOREIGN KEY (account_id) REFERENCES accounts(account_id) ON DELETE CASCADE,
  FOREIGN KEY (address_id) REFERENCES addresses(address_id) ON DELETE CASCADE
);

CREATE TABLE account_phones
(
  account_id INT NOT NULL,
  phone_id INT NOT NULL,
  PRIMARY KEY (account_id, phone_id),
  FOREIGN KEY (account_id) REFERENCES accounts(account_id) ON DELETE CASCADE,
  FOREIGN KEY (phone_id) REFERENCES phone_numbers(phone_id) ON DELETE CASCADE
);

CREATE TABLE account_geos
(
  account_id INT NOT NULL,
  geo_id INT NOT NULL,
  PRIMARY KEY (account_id, geo_id),
  FOREIGN KEY (account_id) REFERENCES accounts(account_id) ON DELETE CASCADE,
  FOREIGN KEY (geo_id) REFERENCES geolocations(geo_id) ON DELETE CASCADE
);

CREATE TABLE account_ips
(
  account_id INT NOT NULL,
  ip_id INT NOT NULL,
  PRIMARY KEY (account_id, ip_id),
  FOREIGN KEY (account_id) REFERENCES accounts(account_id) ON DELETE CASCADE,
  FOREIGN KEY (ip_id) REFERENCES ip_addresses(ip_id) ON DELETE CASCADE
);

CREATE TABLE account_devices
(
  account_id INT NOT NULL,
  device_id INT NOT NULL,
  PRIMARY KEY (account_id, device_id),
  FOREIGN KEY (account_id) REFERENCES accounts(account_id) ON DELETE CASCADE,
  FOREIGN KEY (device_id) REFERENCES devices(device_id) ON DELETE CASCADE
);

CREATE TABLE transactions
(
  transaction_id BIGSERIAL PRIMARY KEY,
  account_id INT NOT NULL,
  from_account_id INT NOT NULL,
  to_account_id INT NOT NULL,
  encrypted_amount BYTEA NOT NULL,
  is_fraud BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  CONSTRAINT fk_account
    FOREIGN KEY (account_id)
    REFERENCES accounts(account_id)
    ON UPDATE cascade ON DELETE NO ACTION,
  CONSTRAINT fk_from_account
    FOREIGN KEY (from_account_id)
    REFERENCES accounts(account_id)
    ON UPDATE cascade ON DELETE NO ACTION,
  CONSTRAINT fk_to_account
    FOREIGN KEY (to_account_id)
    REFERENCES accounts(account_id)
    ON UPDATE cascade ON DELETE NO ACTION
);

CREATE TABLE transactions_encrypted
(
  transaction_id BIGSERIAL PRIMARY KEY,
  account_id INT NOT NULL,
  amount NUMERIC(15,2) NOT NULL,
  encrypted_data BYTEA NOT NULL,
  created_date TIMESTAMP NOT NULL DEFAULT now(),
  from_account_id INT NOT NULL,
  to_account_id INT NOT NULL,
  is_fraud BOOLEAN NOT NULL DEFAULT false,
  CONSTRAINT fk_accounts
    FOREIGN KEY (account_id)
    REFERENCES accounts(account_id)
    ON UPDATE cascade ON DELETE NO ACTION,
  CONSTRAINT fk_from_account
    FOREIGN KEY (from_account_id)
    REFERENCES accounts(account_id)
    ON UPDATE cascade ON DELETE NO ACTION,
  CONSTRAINT fk_to_account
    FOREIGN KEY (to_account_id)
    REFERENCES accounts(account_id)
    ON UPDATE cascade ON DELETE NO ACTION
);

CREATE TABLE transaction_keys
(
  transaction_id INT PRIMARY KEY
    REFERENCES transactions_encrypted(transaction_id) ON DELETE NO ACTION,
  bb84_key BYTEA NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE transactions_qml
(
  transaction_id SERIAL PRIMARY KEY,
  amount NUMERIC(19,4) NOT NULL,
  transaction_time TIMESTAMP NOT NULL,
  geo_location TEXT,
  ip_address TEXT,
  is_fraud BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE algorithms
(
  id SERIAL PRIMARY KEY,
  algorithm_name TEXT
);

INSERT INTO algorithms
  (algorithm_name)
VALUES
  ('qsvm'),
  ('vqc');

CREATE TABLE classification_metrics
(
  id SERIAL PRIMARY KEY,
  algorithm_type_id INT NOT NULL,
  accuracy DOUBLE PRECISION NOT NULL,
  "precision" DOUBLE PRECISION NOT NULL,
  recall DOUBLE PRECISION NOT NULL,
  f1_score DOUBLE PRECISION NOT NULL,
  confusion_matrix JSONB NOT NULL,
  roc_curve JSONB NOT NULL,
  CONSTRAINT fk_algorithm_type
    FOREIGN KEY (algorithm_type_id)
    REFERENCES algorithms(id)
    ON UPDATE cascade
);