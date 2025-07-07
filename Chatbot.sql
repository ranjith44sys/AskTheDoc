create database chatbot;
use chatbot;
SET SQL_SAFE_UPDATES = 0;
create table uploaded_file (
 id INT AUTO_INCREMENT PRIMARY KEY,
 filename varchar(255),
 data LONGBLOB
 );


CREATE TABLE  faiss_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255),
    index_blob LONGBLOB,
    metadata_json LONGTEXT
);
CREATE TABLE  search_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query TEXT NOT NULL,
    result_json LONGTEXT NOT NULL,
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE  search_history_1 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query TEXT NOT NULL,
    result_json LONGTEXT NOT NULL,
    llm_answer LONGTEXT,
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE  comments (
        id INT AUTO_INCREMENT PRIMARY KEY,
        comment TEXT,
        rating FLOAT,
        polarity FLOAT,
        offensive BOOLEAN
    );

 select * from uploaded_file;
 select * from search_history;
select * from search_history_1;
select * from comments;