# Classification and Categorization of Short Text

## Data Format
Each file is a JSON document, containing a flat list of joke objects. Each joke object always has a `body`, `category`, and `id` field with other fields varying between sets. 

### stupidstuff.json
Scraped from [stupidstuff.org](stupidstuff.org/jokes/).

Additional fields:

* `id` -- page ID on stupidstuff.org.
* `category` -- see available categories [here](http://stupidstuff.org/jokes/category.htm).
* `rating` -- mean user rating on a scale of 1 to 5.

```json
{
        "category": "Blonde Jokes",
        "body": "A blonde is walking down the street with her blouse open, exposing one of her breasts. A nearby policeman approaches her and remarks, \"Ma'am, are you aware that I could cite you for indecent exposure?\" \"Why, officer?\" asks the blonde. \"Because your blouse is open and your breast is exposed.\" \"Oh my goodness,\" exclaims the blonde, \"I must have left my baby on the bus!\"",
        "id": 14,
        "rating": 3.5
    }
```


### wocka.json
Scraped from [wocka.com](http://wocka.com/).

Additional fields:

* `id` -- page ID on wocka.com.
* `category` -- see available categories [here](http://www.wocka.com/).
* `title` -- title of the joke.

```json
{
        "title": "Infants vs Adults",
        "body": "Do infants enjoy infancy as much as adults enjoy adultery?",
        "category": "One Liners",
        "id": 17
    }
```


## Code Structure
Our code takes in both datasets, 
removes all columns other than the `body` and `category` columns, and 
removes all categories not shared between the two datasets. It then 
maps each category to a new column `category_code` where each category 
label is assigned a numerical value for processing. The raw text data in each row is cleaned and then the `body` and 
`category_code` columns are split into training and testing, then vectorized in a TF-IDF vectorizer.
 
For the joke recommendation - the user is asked to input a joke, the code then cleans the users input.  Using logistic 
regression, it predicts the `category_code` of the inputted joke and identifies a joke similar to the input, using cosine similarity, 
and prints it to the screen. 

## How to compile, set up, and deploy the system
To begin, drag and drop both the wocka.json and stupidstuff.json files into the folder that contains the python file. 
Do not change the names of the files -- if you do, change lines 36 and 37 of the python file to match the new names. 
Run the python file, enter your input, and expect results. 

If you wish to see the performance of each model, you can uncomment the code on the bottom and run it.  It will print out
10-fold cross validation scores, the mean and standard deviation of the scores, and a classification report of the model.


## Limitations
The code will several minutes to fully run and build each model.  


