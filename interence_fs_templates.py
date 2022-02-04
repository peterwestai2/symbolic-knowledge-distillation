'''

Template Definition

'''




'''

ADD xWant

'''

'''

ADD xWant

'''

##### DEFINE THIS ######
relation_to_test = 'xWant'
########################

examples_string = """PersonX is at a party	xWant	to drink beer and dance
PersonX bleeds a lot	xWant	to see a doctor
PersonX works as a cashier	xWant	to be a store manager
PersonX gets dirty	xWant	to clean up
PersonX stays up all night studying	xWant	to sleep all day
PersonX gets PersonY's autograph	xWant	to have a relationship with PersonY
PersonX ends a friendship	xWant	to meet new people
PersonX makes his own costume	xWant	to go to a costume party
PersonX calls PersonY	xWant	to have a long chat
PersonX tells PersonY a secret	xWant	to get PersonY's advice
PersonX mows the lawn	xWant	to get a new lawnmower
PersonX makes a huge mistake	xWant	to apologize
PersonX sees PersonY's point	xWant	to agree with PersonY
PersonX wants a tattoo	xWant	to find a tattoo design
PersonX tells PersonY something	xWant	to get PersonY's opinion
PersonX leaves PersonY's bike	xWant	to keep the bike safe
PersonX visits some friends	xWant	to catch up with friends
PersonX sits next to PersonY	xWant	to become better friends with PersonY
PersonX gives PersonY indication	xWant	to date PersonY
PersonX buys some popcorn	xWant	eat popcorn from bag in car"""
    
lines = [v.split('\t') for v in  examples_string.split('\n')]

examples = [{'head':v[0], 'relation':v[1], 'tail':v[2]} for v in lines]

assert(examples[0]['relation'] == relation_to_test)



def ex2text(ex, include_gen = True, number=None):

    text = ''
    
    if number is not None:
        text += 'Situation {}: '.format(number)
    
    text += '{}.\n\nWants: As a result, PersonX wants'.format(ex['head'],ex['head'])

    if include_gen:
        text += ' {}.\n\n'.format(ex['tail'])
    return text



xWant_dict = {'relation':'xWant', 
              'examples':examples[:7],
             'function':ex2text,
             'prefix':'How does this situation affect each character\'s wants?\n\n\n',
             'length':8}

'''

ADD xAttr

'''

##### DEFINE THIS ######
relation_to_test = 'xAttr'
########################

xAttr_examples_string = """PersonX bullies PersonY	xAttr	dominant
PersonX moves to another city	xAttr	adventurous
PersonX changes PersonY's mind	xAttr	persuasive
PersonX writes a story	xAttr	creative
PersonX covers PersonY's expenses	xAttr	generous
PersonX takes time off	xAttr	lazy
PersonX advises PersonY	xAttr	wise
PersonX bursts into tears	xAttr	sensitive
PersonX deals with problems	xAttr	determined
PersonX follows PersonY	xAttr	creepy"""
    
lines = [v.split('\t') for v in  xAttr_examples_string.split('\n')]

xAttr_examples = [{'head':v[0], 'relation':'xAttr', 'tail':v[2]} for v in lines]



def ex2text(ex, include_gen = True, number=None):

    text = ''
    
    if number is not None:
        text += '{}. '.format(number)
    
    text += 'When {}, people think PersonX is'.format(ex['head'])

    if include_gen:
        text += ' {}.\n\n'.format(ex['tail'])
    return text




xAttr_dict = {'relation':'xAttr', 
              'examples':xAttr_examples[:10],
             'function':ex2text,
             'prefix':'Next, we will discuss how people are seen in different situations. Examples:\n\n\n',
             'length':5}



'''

ADD xIntent

'''

##### DEFINE THIS ######
relation_to_test = 'xIntent'
########################



examples_string = """PersonX gets the newspaper	xIntent	to read the newspaper
PersonX works all night	xIntent	to meet a deadline
PersonX destroys PersonY	xIntent	to punish PersonY
PersonX clears her mind	xIntent	to be ready for a new task
PersonX wants to start a business	xIntent	to be self sufficient
PersonX ensures PersonY's safety	xIntent	to be helpful
PersonX buys lottery tickets	xIntent	to become rich
PersonX chases the ball	xIntent	to return it to some kids
PersonX calls customer service	xIntent	to solve a problem
PersonX hops around	xIntent	to exercise"""
    
    
lines = [v.split('\t') for v in  examples_string.split('\n')]

examples = [{'head':v[0], 'relation':v[1], 'tail':v[2]} for v in lines]


def ex2text(ex, include_gen = True, number=None):
    text = ''
    
    if number is not None:
        text += '{}. '.format(number)
    #text += '\n\nEvent: {}\n\nCharacter: PersonX\n\nIntention: "{}" because PersonX is trying'.format(ex['head'],ex['head'])
    text += '\n\nEvent: {}.\n\nIntention: "{}" because PersonX wants'.format(ex['head'],ex['head'])

    if include_gen:
        text += ' {}.\n\n'.format(ex['tail'])
    return text



xIntent_dict = {'relation':relation_to_test, 
              'examples':examples[:5],
             'function':ex2text,
               'prefix':'For each event, what do the characters want?\n\n\n'}


'''

ADD xEffect

'''

'''

ADD xEffect

'''

##### DEFINE THIS ######
relation_to_test = 'xEffect'
########################



examples_string = """PersonX recently divorced	xEffect	is single
PersonX lifts weights	xEffect	gains muscle mass
PersonX takes PersonY to a bar	xEffect	gets drunk
PersonX decides to hire a tutor	xEffect	learns to read
PersonX buys PersonY a gift	xEffect	is thanked by PersonY
PersonX hears bad news	xEffect	faints
PersonX buys something	xEffect	gets change
PersonX does a lot of work	xEffect	gets better at their job
PersonX attends the concert	xEffect	enjoys the music
PersonX performs PersonX's duties	xEffect	receives the expected salary
PersonX gets calls	xEffect	has a conversation
PersonX continues PersonY's search	xEffect	gets lost
PersonX sees logs	xEffect	calms down"""
    
lines = [v.split('\t') for v in  examples_string.split('\n')]

examples = [{'head':v[0], 'relation':v[1], 'tail':v[2]} for v in lines]

assert(examples[0]['relation'] == relation_to_test)



def ex2text(ex, include_gen = True):
    
    text = 'Because {},{}'.format(ex['head'], 
                                 ' she feels')
    if include_gen:
        text += ' {}\n#\n'.format(ex['tail'])
    return text


def ex2text(ex, include_gen = True, number=None):

    
    text = ''
    
    if number is not None:
        text += '{}.\n\n'.format(number)
    
    text += 'event: {}\n\noutcome: PersonX'.format(ex['head'],ex['head'])

    if include_gen:
        text += ' {}\n\n'.format(ex['tail'])
    return text



xEffect_dict = {'relation':relation_to_test, 
              'examples':examples[:10],
             'function':ex2text,
              'prefix':'What is the result?\n\n\n'}








'''

ADD xReact


'''

##### DEFINE THIS ######
relation_to_test = 'xReact'
########################



examples_string = """PersonX lives with PersonY's family	xReact	loved
PersonX expects to win	xReact	excited
PersonX comes home late	xReact	tired
PersonX sees dolphins	xReact	joyful
PersonX causes PersonY anxiety	xReact	guilty
PersonX goes broke	xReact	embarrassed
PersonX has a drink	xReact	refreshed
PersonX has a heart condition	xReact	scared about their health
PersonX shaves PersonY's hair	xReact	helpful
PersonX loses all of PersonY's money	xReact	horrible
PersonX finishes dinner	xReact	relaxed and content
PersonX decides to go see a movie	xReact	immersed in a different reality
PersonX comes home early	xReact	happy to be not at work
PersonX finds peace	xReact	in a position of grace
PersonX says anything else	xReact	glad that people listened to him"""
    
lines = [v.split('\t') for v in  examples_string.split('\n')]

examples = [{'head':v[0], 'relation':v[1], 'tail':v[2]} for v in lines]

assert(examples[0]['relation'] == relation_to_test)

##### DEFINE THIS ######
relation_to_test = 'xReact'
########################



examples_string = """PersonX lives with PersonY's family	xReact	loved
PersonX expects to win	xReact	excited
PersonX comes home late	xReact	tired
PersonX has a drink	xReact	refreshed
PersonX sees dolphins	xReact	joyful
PersonX causes PersonY anxiety	xReact	guilty
PersonX goes broke	xReact	embarrassed
PersonX has a drink	xReact	refreshed
PersonX has a heart condition	xReact	scared about their health
PersonX shaves PersonY's hair	xReact	helpful
PersonX loses all PersonY's money	xReact	bad they lost someone's belongings
PersonX finishes dinner	xReact	relaxed and content
PersonX decides to go see a movie	xReact	immersed in a different reality
PersonX comes home early	xReact	happy to be not at work
PersonX finds peace	xReact	in a position of grace
PersonX says anything else	xReact	glad that people listened to him"""
    
lines = [v.split('\t') for v in  examples_string.split('\n')]

examples = [{'head':v[0], 'relation':v[1], 'tail':v[2]} for v in lines]

assert(examples[0]['relation'] == relation_to_test)

def ex2text(ex, include_gen = True):
    
    text = 'Because {},{}'.format(ex['head'], 
                                 ' she feels')
    if include_gen:
        text += ' {}\n#\n'.format(ex['tail'])
    return text


def ex2text(ex, include_gen = True, number=None):

    
    text = ''
    
    if number is not None:
        text += '{}.\n\n'.format(number)
    
    text += 'Event: {}\n\nOutcome: PersonX feels'.format(ex['head'])

    if include_gen:
        text += ' {}\n\n'.format(ex['tail'])
    return text



xReact_dict = {'relation':relation_to_test, 
              'examples':examples[:10],
             'function':ex2text,
              'prefix':'How do the characters react to the event?\n\n\n'}






def ex2text(ex, include_gen = True):
    
    text = 'Because {},{}'.format(ex['head'], 
                                 ' she feels')
    if include_gen:
        text += ' {}\n#\n'.format(ex['tail'])
    return text


def ex2text(ex, include_gen = True, number=None):

    
    text = ''
    
    if number is not None:
        text += 'Situation {}: '.format(number)
    
    text += '{}.\n\nPersonX feels:'.format(ex['head'])

    if include_gen:
        text += ' {}.\n\n'.format(ex['tail'])
    return text



xReact_dict = {'relation':relation_to_test, 
              'examples':examples[:10],
             'function':ex2text,
              'prefix':'Next, how do people feel in each situation? Examples:\n\n\n',
              'length':8}






'''

ADD xNeed

'''

'''

ADD xNeed

'''

##### DEFINE THIS ######
relation_to_test = 'xNeed'
########################



examples_string = """PersonX gets a job offer	xNeed	to apply
PersonX eats the food	xNeed	to be hungry
PersonX gets a date	xNeed	to ask someone out
PersonX has a baby shower	xNeed	to invite people
PersonX makes many new friends	xNeed	to be social
PersonX changes PersonY's mind	xNeed	to make PersonY think about the issue
PersonX watches Netflix	xNeed	to have a Netflix account
PersonX rides PersonY's skateboard	xNeed	to be at the skate park
PersonX takes a quick nap	xNeed	to lie down
PersonX handles the situation	xNeed	to have an issue
PersonX gets a new phone	xNeed	to pay for the new phone
PersonX takes the trash out	xNeed	to gather the trash
"""
    
lines = [v.split('\t') for v in  examples_string.split('\n')]

examples = [{'head':v[0], 'relation':v[1], 'tail':v[2]} for v in lines]

assert(examples[0]['relation'] == relation_to_test)


def ex2text(ex, include_gen = True, number = None):

    text = ''
    if number is not None:
        text += 'Event {}.: '.format(number)
    text += '{}\n\nPrerequisites: For this to happen, PersonX needed'.format(ex['head'],ex['head'])

    if include_gen:
        text += ' {}.\n\n'.format(ex['tail'])
    return text



xNeed_dict = {'relation':relation_to_test, 
              'examples':examples[:10],
             'function':ex2text,
             'prefix':'What needs to be true for this event to take place?\n\n\n'}








'''

ADD HinderedBy

TODO: select examples

'''

##### DEFINE THIS ######
relation_to_test = 'HinderedBy'
########################

examples_string = """PersonX makes a doctor's appointment	HinderedBy	PersonX can't find the phone to call the doctor
PersonX rubs PersonY's forehead	HinderedBy	PersonX is afraid to touch PersonY
PersonX eats peanut butter	HinderedBy	PersonX is allergic to peanuts
PersonX looks perfect	HinderedBy	PersonX can\'t find any makeup
PersonX goes on a run	HinderedBy	PersonX has a knee injury
PersonX takes PersonY to the emergency room	HinderedBy	PersonY has no health insurance to pay for medical care
PersonX spends time with PersonY's family	HinderedBy	PersonY’s family doesn't like spending time with PersonX
PersonX moves from place to place	HinderedBy	PersonX can\'t afford to move
PersonX protests the government	HinderedBy	PersonX is arrested
PersonX has a huge fight	HinderedBy	PersonX does not like confrontation
PersonX understands PersonY's feelings	HinderedBy	PersonY will not talk to PersonX
PersonX asks PersonY a question	HinderedBy	PersonY cannot hear PersonX
PersonX loves PersonY very much	HinderedBy	PersonY is mean to PersonX
PersonX makes a request	HinderedBy	PersonX is afraid of being rejected
PersonX meets PersonY's spouse	HinderedBy	PersonY's spouse is out of the country
PersonX fills a waterbottle	HinderedBy	There is no sink nearby to fill it
PersonX schedules an appointment	HinderedBy	The receptionist won't answer PersonX's calls
PersonX becomes rich	HinderedBy	PersonX has no revenue
PersonX gets the job done	HinderedBy	PersonX is too tired to work
PersonX sits alone in a room	HinderedBy	People come into the room and bother PersonX
PersonX evaluates PersonY’s performance	HinderedBy	PersonX misses PersonY's performance
PersonX mows lawns	HinderedBy	PersonX's lawn mower breaks"""
    
lines = [v.split('\t') for v in  examples_string.split('\n')]

examples = [{'head':v[0], 'relation':v[1], 'tail':v[2]} for v in lines]

assert(examples[0]['relation'] == relation_to_test)



## converts an example into text for few-shot
def ex2text(ex, include_gen = True, number = None):
    text = ''
    if number is not None:
        text += '{}. Event: '.format(number)
        
    #text += '"{}"\n\nCondition: "{}" will not occur under the condition that'.format(ex['head'],ex['head'],number)
    text += '{}.\n\nCondition: "{}" is untrue because'.format(ex['head'],ex['head'],number)

    if include_gen:
        text += ' {}.\n\n\n'.format(ex['tail'])
    return text


## dictionary defining few-shot template for HinderedBy relation
## prefix will be put at the beginning, before few-shot examples
HinderedBy_dict = {'relation':relation_to_test, 
              'examples':examples[:5],
             'function':ex2text,
                  'prefix': 'What condition stops this event from occurring?\n\n\n'}


relation_dicts = (HinderedBy_dict,
                 xNeed_dict, 
                 xWant_dict,
                 xIntent_dict,
                 xReact_dict,
                 xAttr_dict,
                 xEffect_dict)