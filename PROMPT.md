## Task Description
Create a conversational agent (using Python code) that can leverage multiple
sources of product information to make informed recommendations via a chat
interface. The agent should take as input customer messages. Optionally,
incorporate an understanding of images as well as text for multi modal input. The
output should be text with clarification questions or relevant product
recommendations (include the article id). Focus on complex conversation
scenarios with multiple messages.
Requirements:
Only recommend relevant products based on the customer conversation
The agent should ask clarification questions to understand user requirements
We are particularly interested in how you think about designing a robust
conversational system, not just retrieving results. Strong solutions typically
demonstrate thoughtful handling of:
Ambiguous or underspecified user queries
When and how to ask clarification questions
Maintaining context across multiple turns
Failure modes (e.g. no results, low-confidence matches) and recovery
strategies
Out-of-scope or unsafe requests
These are not strict requirements, but areas where we expect to see clear
reasoning and design decisions.
We are looking for creative solutions! A non-exhaustive list of optional
ideas/examples:
- Multi-Agent implementation
- Cite the information the agent used to answer a question
- Expand the articles.csv dataset by analysing the associated images.
- Automated performance evaluation of the agent
- Automated creation and optimisation of prompts
- Use external sources of information by searching the web
- Use some guardrail technique to prevent the agent from answering out-of-
scope questions

Conversation Examples:
Example Use Case - Product Advice
[User]
Do you have the Howie shorts in stock in blue?
[Agent]
The Howie shorts are a great choice!
We have a dark turquoise colour in stock.
**provide Howie Shorts in dark turquoise product id**

Example Use Case - Guiding the customer through conversation
[User]
Hey, I am interested in buying some clothes
[Agent]
Hello, I’d be happy to assist you in finding the right produc
ts.
Could you please provide some more information?
Is there a special event or activity you want to find an outf
it for?
[User]
I just want a new outfit for this weekend
[Agent]
We have a variety of clothing in stock, shall we start by get
ting you the right trousers, top and jacket combination?
[User]
That sounds good, although it’s still quite warm, so shorts a
nd a blouse will be enough.
[Agent]
We have a variety of patterns and colours in our collection o
f shorts.
Do you prefer a solid colour, denim, striped, all-over patter
n?
Do you have any similar preferences for a blouse?
For example, a lighter-coloured blouse could work well with d
enim shorts.
[User]
The denim shorts sound good, maybe with a white blouse?
[Agent]
Great choice!
Here is the denim shorts and white blouse combination for you
r weekend.
**provide a list of relevant products ids**

Example Use Case - Find garments based on text + image
[User]
I saw this picture on Instagram and I want the trousers she i
s wearing. What would you suggest?
**image attachment**
[Agent]
We have similar red trousers in stock, let me know if you pre
fer these.
**provide a list of relevant products ids**

### Project Submission
Submit all project files necessary to run your project in a zip file or Github
repository. Include requirements.txt for Python dependencies.
If you train any models, include model weights and supporting files. Include all
the datasets you’ve used/changed.
Include a text document with your thought process and the steps you’ve taken
to implement the solution.

### Data
articles.csv consists of a list of Women’s products.
Each article/item has a free text title and description and some other categorical
information. Most of the articles also have associated product images in the
images directory. You are free to transform the data points or use additional
datasets to support your ML methods.