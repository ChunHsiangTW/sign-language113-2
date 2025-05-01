SHOW DATABASES;
USE hand_project;
SHOW TABLES;
CREATE TABLE `users`(
	`user_id` INT AUTO_INCREMENT PRIMARY KEY,
    `name` VARCHAR(100) NOT NULL,
	`gender` ENUM('M', 'F'),
	`age` INT,
    `email` VARCHAR(255) UNIQUE NOT NULL,
    `password` VARCHAR(255) NOT NULL
);

CREATE TABLE `courses`(
	`course_id` INT AUTO_INCREMENT PRIMARY KEY,
    `title` VARCHAR(50) UNIQUE NOT NULL,
	`level` ENUM('Beginner', 'Intermediate', 'Expert'),
	`desc` TEXT,
    `thumbnail_url` VARCHAR(255)
);
