import openai 


def scrub_PX_PY(s, PersonX, PersonY):
    return s.replace(PersonX, 'PersonX').replace(PersonY, 'PersonY')

def name_PX_PY(s, PersonX, PersonY):
    return s.replace('PersonX', PersonX).replace('PersonY', PersonY)

def complete_gpt3(prompt, l, model_name, num_log_probs=None,  n=None, top_p = 0.5, stop = '\n', echo=True,
                 frequency_penalty = 0., presence_penalty=0. ):
    global total_cost
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(engine=model_name, prompt=prompt, max_tokens=l, temperature=1.,
                                                logprobs=num_log_probs, echo=echo, stop=stop, n=n, top_p=float(top_p),
                                               frequency_penalty = frequency_penalty, 
                                                presence_penalty= presence_penalty)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False

            print("API error:", error)
            time.sleep(0.2)
            
    #total_cost += engine_price[model_name]*get_tokens_consumed(response)
    return response

## create a few-shot prompt with the final 
## example left open for generation
##
### note: ex2text should be a function that takes 
## an example from examples and produces a string 
## template. This should accept an argument, include_gen 
## which is a bool to decide whether to leave it open for
## generation or include the gt
def few_shot_prompt(examples, ex2text, number = None, Person_list = None):
    template_str = ''
    
    i = -1
    for i, example in enumerate(examples[:-1]):
        
        if number:
            ex_str = ex2text(example, include_gen = True, number = i + 1)
        else:
            ex_str = ex2text(example, include_gen = True)
            
            
        if Person_list is not None:
            ex_str = name_PX_PY(ex_str, Person_list[i][0],Person_list[i][1] )
            
        template_str += ex_str
        
    i = i + 1
    
    if number:
        ex_str = ex2text(examples[-1], include_gen = False,number = i + 1)
    else:
        ex_str = ex2text(examples[-1], include_gen = False)
        
    if Person_list is not None:
        ex_str = name_PX_PY(ex_str, Person_list[i][0],Person_list[i][1] )
    
    template_str += ex_str
    
    return template_str