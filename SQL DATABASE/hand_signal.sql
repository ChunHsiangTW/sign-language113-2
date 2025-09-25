SHOW DATABASES;
USE hand_project; 
SHOW TABLES;

CREATE TABLE `users`(
	`user_id` INT AUTO_INCREMENT PRIMARY KEY,
    `name` VARCHAR(100) NOT NULL,
	`gender` ENUM('M', 'F', 'Other'),
	`age` INT,
    `email` VARCHAR(255) UNIQUE NOT NULL,
    `password` VARCHAR(255) NOT NULL
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





    
    
