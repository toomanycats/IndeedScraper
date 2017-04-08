USE indeed;

CREATE TABLE data ( 
    ENGINE=InnoDB, 
    AUTO_INCREMENT=1,
    DEFAULT CHARSET=Latin1,
    `keyword` varchar(100) UNIQUE NOT NULL,
    `df_file` varchar(50) UNIQUE NOT NULL,
    `type_` ENUM('keyword', 'title') NOT NULL,
    `session_id` VARCHAR(36) UNIQUE NOT NULL,
    `ind` MEDIUMINT(9) UNSIGNED DEFAULT NULL,
    `end` SMALLINT(6) UNSIGNED DEFAULT NULL,
    `count_thres` SMALLINT(6) UNSIGNED DEFAULT NULL,
    `stem_inv` TEXT,
    `bigram` TEXT,
    `pri_key` MEDIUMINT(9) UNSIGNED NOT NULL AUTO_INCREMENT,
    PRIMARY KEY (pri_key)
    ) 


