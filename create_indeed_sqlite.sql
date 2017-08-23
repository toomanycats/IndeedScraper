create table data
(keyword      varchar(100) not null,
 df_file      varchar(255),
 type_        varchar(10),
 session_id   varchar(36),
 ind          mediumint(9),
 end          smallint(6),
 count_thres  smallint(6),
 stem_inv     text,
 bigram       text,
 pri_key      integer unsigned  auto_increment primary key
);

