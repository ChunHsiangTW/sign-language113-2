SHOW DATABASES;
USE `114-Handlang`;
SHOW TABLES;

CREATE TABLE `users`(
	`id` INT AUTO_INCREMENT PRIMARY KEY,
    `name` VARCHAR(50) UNIQUE NOT NULL,
    `age` INT,
    `email` VARCHAR(255) UNIQUE NOT NULL,
    `gender` ENUM('M', 'F', 'Other'),
    `password` VARCHAR(255) NOT NULL
);
    
CREATE TABLE `category`(
	`category_id` INT AUTO_INCREMENT PRIMARY KEY,
    `category_title` VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE `gesture_video`(
	`video_id` INT PRIMARY KEY AUTO_INCREMENT,
    `category_id` INT NOT NULL,
    `word` VARCHAR(50) NOT NULL,
    `video_url` VARCHAR(255),
    FOREIGN KEY (category_id) REFERENCES category(category_id)
);

CREATE TABLE `recognize_record`(
	`record_id` INT PRIMARY KEY AUTO_INCREMENT,
    `id` INT NOT NULL,
    `recognized_text` VARCHAR(500) NOT NULL,
    `recognized_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id) REFERENCES users(id)
);
