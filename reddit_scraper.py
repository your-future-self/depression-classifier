import praw
import VALUES as V
def main():
    reddit = praw.Reddit(
        client_id= V.CLIENT_ID,
        client_secret= V.CLIENT_SECRET,
        password= V.PASSWORD,
        user_agent= V.USER_AGENT,
        username= V.USERNAME,
    )
    subreddit_list = ['nyc', 'losangeles','chicago','dallas',
    'houston', 'washingtondc', 'philadelphia', 'atlanta', 'miami','phoenix',
    'boston', 'sanfrancisco', 'Detroit','Seattle', 'Minneapolis',]
    
    short_list =  ['losangeles','chicago','dallas', 'washingtondc', 'philadelphia', 'atlanta']
    
    for city in short_list:
        city_subreddit = reddit.subreddit(city)
        titles_and_comments(city_subreddit)

  


def titles_and_comments(sub_reddit):
    """
    Takes in a subreddit of reddit posts and puts 
    """
    posts = sub_reddit.top(time_filter = 'year', limit = 25)
    title = sub_reddit.display_name
    with open(f'{title}_titles_and_comments.txt', 'w') as sink:
        for post in posts:
                sink.write(post.title)
                for comment in post.comments.list(): #cleaning mlsoreComments could probably be turned into a function later
                    if type(comment) != praw.models.MoreComments:
                        sink.write(comment.body)



        


if __name__ == '__main__':
    main()