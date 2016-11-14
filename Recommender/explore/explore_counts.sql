-- Count the number of occurrences of each actor
select actorID, count(actorID) as occurrence from movie_actors
group by actorID
order by count(actorID) desc

 -- Count the number of occurences of each director
select directorID, count(directorID) as occurrence from movie_directors
group by directorID
order by count(directorID) desc 

 -- Count the number of movies in each genre
select genre, count(genre) as genre_count from movie_genres
group by genre
order by count(genre) desc
