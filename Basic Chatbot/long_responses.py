import random

R_EATING = "Are you DUMB? Can't you see that I'm a bot, I don't eat anything you DUMBFUCK!"

def unknown():
    response = ['Could you please re_phrase that?',
                'Wie bitte?',
                "Huh Wut?",
                "What does that mean?",
                "Sorry I don't know the answer to your question :("
                "HAR?"][random.randrange(4)]
    return response
