select user_id, avg(rating) from train
group by user_id 
order by user_id asc

select avg(rating) as avg_rating from train

select movieID, avg(rating) as avg_movie_rating from train 
group by movieID
order by movieID asc