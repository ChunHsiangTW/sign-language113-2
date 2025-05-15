SHOW DATABASES;
USE hand_project; 
SHOW TABLES;

CREATE TABLE `users`(
	`user_id` INT AUTO_INCREMENT PRIMARY KEY,
    `name` VARCHAR(100) NOT NULL,
	`gender` ENUM('M', 'F', 'Other'),
	`age` INT,
    `bio` TEXT,
    `email` VARCHAR(255) UNIQUE NOT NULL,
    `password` VARCHAR(255) NOT NULL,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE `user_providers`(
	`id` INT PRIMARY KEY AUTO_INCREMENT,
    `user_id` INT,
    `provider` ENUM('google', 'facebook', 'apple'),
    `provider_uid` VARCHAR(255),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE `email_tokens` (
	`id` INT PRIMARY KEY AUTO_INCREMENT,
	`user_id` INT,
	`token` VARCHAR(255),
	`type` ENUM('verify', 'reset'),
	FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE `courses`(
	`course_id` INT AUTO_INCREMENT PRIMARY KEY,
    `course_title` VARCHAR(50) UNIQUE NOT NULL,
	`level` ENUM('Beginner', 'Intermediate', 'Expert'),
	`desc` TEXT,
    `thumbnail_url` VARCHAR(255)
);

CREATE TABLE `lessons`(
	`lesson_id` INT AUTO_INCREMENT PRIMARY KEY,
	`course_id` INT,
    `lesson_title` VARCHAR(50) UNIQUE NOT NULL,
	`video_url` VARCHAR(255),
	`content` TEXT,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE
);

CREATE TABLE `gesture_word`(
	`gesture_id` INT AUTO_INCREMENT PRIMARY KEY,
    `word` VARCHAR(50) NOT NULL,
    `meaning` TEXT,
    `example` TEXT,
    `category` VARCHAR(50)
);

CREATE TABLE `gesture_video`(
	`video_id` INT PRIMARY KEY AUTO_INCREMENT,
    `gesture_id` INT NOT NULL,
    `video_url` VARCHAR(255),
    FOREIGN KEY (gesture_id) REFERENCES gesture_word(gesture_id)
);

CREATE TABLE `sign_category`(
	`category_id` INT PRIMARY KEY AUTO_INCREMENT,
    `category_name` VARCHAR(50) NOT NULL
);

CREATE TABLE `user_goals` (
	`user_id` INT PRIMARY KEY,
	`minutes_per_day` INT DEFAULT 10,
	`start_date` DATE,
	`last_updated` DATETIME DEFAULT CURRENT_TIMESTAMP,
	FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE `user_learning_logs` (
	`id` INT PRIMARY KEY AUTO_INCREMENT,
	`user_id` INT,
	`word_id` INT,
	`studied_at` DATETIME,
	`duration_seconds` INT,
	FOREIGN KEY (user_id) REFERENCES users(user_id),
	FOREIGN KEY (word_id) REFERENCES gesture_word(gesture_id)
);

CREATE TABLE `user_badges` (
  `user_id` INT,
  `badge_code` VARCHAR(50),
  `earned_at` DATETIME,
  PRIMARY KEY (user_id, badge_code),
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);




    
    
