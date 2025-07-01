-- 0) Tear everything down
DROP VIEW    IF EXISTS account_profiles
CASCADE;
DROP TYPE    IF EXISTS account_type_enum
CASCADE;

DROP TABLE   IF EXISTS classification_metrics
CASCADE;
DROP TABLE   IF EXISTS transaction_crypto
CASCADE;
DROP TABLE   IF EXISTS key_management
CASCADE;
-- (remove transaction_keys entirely)
DROP TABLE   IF EXISTS transactions_encrypted
CASCADE;
DROP TABLE   IF EXISTS transactions_qml
CASCADE;
DROP TABLE   IF EXISTS transactions
CASCADE;

DROP TABLE   IF EXISTS account_devices
CASCADE;
DROP TABLE   IF EXISTS account_ips
CASCADE;
DROP TABLE   IF EXISTS account_geos
CASCADE;
DROP TABLE   IF EXISTS account_phones
CASCADE;
DROP TABLE   IF EXISTS account_addresses
CASCADE;

DROP TABLE   IF EXISTS devices
CASCADE;
DROP TABLE   IF EXISTS ip_addresses
CASCADE;
DROP TABLE   IF EXISTS geolocations
CASCADE;
DROP TABLE   IF EXISTS phone_numbers
CASCADE;
DROP TABLE   IF EXISTS addresses
CASCADE;

DROP TABLE   IF EXISTS accounts
CASCADE;
DROP TABLE   IF EXISTS algorithms
CASCADE;
DROP TABLE   IF EXISTS roles
CASCADE;
DROP TABLE   IF EXISTS transaction_keys
CASCADE;

-- 1) Roles & Algorithms
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

CREATE TABLE algorithms
(
  id SERIAL PRIMARY KEY,
  algorithm_name TEXT NOT NULL
);
INSERT INTO algorithms
  (algorithm_name)
VALUES
  ('qsvm'),
  ('vqc'),
  ('kyber'),
  ('dilithium'),
  ('aes');

-- 2) Accounts & Contacts
CREATE TABLE accounts
(
  account_id SERIAL PRIMARY KEY,
  encrypted_balance BYTEA NOT NULL,
  username TEXT NOT NULL UNIQUE,
  full_name TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  account_type account_type_enum NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  role_id INT NOT NULL,
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
  phone_number TEXT NOT NULL,
  phone_type TEXT NOT NULL
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

CREATE TABLE account_addresses
(
  account_id INT NOT NULL,
  address_id INT NOT NULL,
  PRIMARY KEY(account_id,address_id),
  FOREIGN KEY(account_id) REFERENCES accounts(account_id) ON DELETE CASCADE,
  FOREIGN KEY(address_id) REFERENCES addresses(address_id) ON DELETE CASCADE
);

CREATE TABLE account_phones
(
  account_id INT NOT NULL,
  phone_id INT NOT NULL,
  PRIMARY KEY(account_id,phone_id),
  FOREIGN KEY(account_id) REFERENCES accounts(account_id) ON DELETE CASCADE,
  FOREIGN KEY(phone_id)   REFERENCES phone_numbers(phone_id) ON DELETE CASCADE
);

CREATE TABLE account_geos
(
  account_id INT NOT NULL,
  geo_id INT NOT NULL,
  PRIMARY KEY(account_id,geo_id),
  FOREIGN KEY(account_id) REFERENCES accounts(account_id) ON DELETE CASCADE,
  FOREIGN KEY(geo_id)     REFERENCES geolocations(geo_id) ON DELETE CASCADE
);

CREATE TABLE account_ips
(
  account_id INT NOT NULL,
  ip_id INT NOT NULL,
  PRIMARY KEY(account_id,ip_id),
  FOREIGN KEY(account_id) REFERENCES accounts(account_id) ON DELETE CASCADE,
  FOREIGN KEY(ip_id)      REFERENCES ip_addresses(ip_id) ON DELETE CASCADE
);

CREATE TABLE account_devices
(
  account_id INT NOT NULL,
  device_id INT NOT NULL,
  PRIMARY KEY(account_id,device_id),
  FOREIGN KEY(account_id) REFERENCES accounts(account_id) ON DELETE CASCADE,
  FOREIGN KEY(device_id)  REFERENCES devices(device_id)   ON DELETE CASCADE
);

-- 3) Transactions & ML stream
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
CREATE TABLE transactions_qml
(
  transaction_id SERIAL PRIMARY KEY,
  amount NUMERIC(19,4) NOT NULL,
  transaction_time TIMESTAMP NOT NULL,
  geo_location TEXT,
  ip_address TEXT,
  is_fraud BOOLEAN NOT NULL DEFAULT FALSE
);

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

-- 4) PQC/AES crypto metadata (no raw BB84)
CREATE TABLE transaction_crypto
(
  transaction_id BIGINT PRIMARY KEY REFERENCES transactions(transaction_id) ON DELETE NO ACTION,
  kyber_ciphertext BYTEA NOT NULL,
  aes_key_wrapped BYTEA NOT NULL,
  key_wrap_kek_id TEXT NOT NULL,
  key_wrap_version TEXT,
  aes_iv BYTEA NOT NULL,
  aes_tag BYTEA,
  dilithium_sig BYTEA NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

-- 5) Centralized KEK store (optional but recommended)
CREATE TABLE key_management
(
  key_id SERIAL PRIMARY KEY,
  kek_id TEXT NOT NULL,
  wrapped BYTEA NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

INSERT INTO classification_metrics
  (algorithm_type_id,accuracy,"precision",recall,f1_score,confusion_matrix,roc_curve)
VALUES
  (1, 0.95, 0.92, 0.90, 0.91, '{"tp":95,"fp":5,"tn":90,"fn":10}', '{"points":[[0,0],[1,1]]}'),
  (2, 0.93, 0.90, 0.88, 0.89, '{"tp":90,"fp":10,"tn":85,"fn":15}', '{"points":[[0,0],[1,1]]}');
