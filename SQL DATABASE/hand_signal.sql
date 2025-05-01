SHOW DATABASES;
USE hand_project;
SHOW TABLES;
DROP TABLE `user_public`;

CREATE TABLE `user_public`(
	`user_id` INT AUTO_INCREMENT PRIMARY KEY,
    `username` VARCHAR(25) NOT NULL,
	`gender` ENUM('M','F'),
    `age` INT
);

DESCRIBE `user_public`;
    
CREATE TABLE `user_private`(
	`user_id` INT PRIMARY KEY,
    `name` VARCHAR(70) NOT NULL,
	`email` VARCHAR(255) NOT NULL,
    `password` VARCHAR(255) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user_public(user_id) ON DELETE CASCADE
);    

INSERT INTO `user_public` (username, gender, age)
VALUES ('alpha', 'M', 24);

SELECT user_id FROM user_public WHERE username = 'Alpha';

INSERT INTO `user_private` (user_id, name, email, password)
VALUES (1, 'Alpha Livingston', 'alpha@gmail.com', 'imd123');

SELECT * FROM `user_private`;

SELECT u.user_id, u.username, u.gender, u.age, p.name, p.email
FROM user_public u
JOIN user_private p ON u.user_id = p.user_id;

SELECT * FROM user_private