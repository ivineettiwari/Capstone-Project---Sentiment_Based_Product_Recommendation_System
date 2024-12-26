# Importing necessary modules
from flask import Flask, render_template, request  # Flask for web app, render_template for rendering HTML templates, request for handling form data
import model  # Importing the custom model module (assumed to contain recommendation logic)

# Initializing Flask application
app = Flask(__name__)  # Use __name__ as the Flask app name (fixed incorrect '__name__' string)

# List of valid user IDs
valid_userid = [
    '00sab00', '1234', 'zippy', 'zburt5', 'joshua', 'dorothy w',
    'rebecca', 'walker557', 'samantha', 'raeanne', 'kimmie',
    'cassie', 'moore222'
]

# Route to render the home page
@app.route('/')
def view():
    """
    Displays the home page where the user can input their user name.
    """
    return render_template('index.html')

# Route to handle product recommendations
@app.route('/recommend', methods=['POST'])
def recommend_top5():
    """
    Handles the recommendation logic. Accepts the user's name via POST request,
    validates the user, and displays the top 5 product recommendations if the user is valid.
    """
    print(request.method)  # Debugging: print the request method
    user_name = request.form['User Name']  # Get the user name from the form input
    print('User name =', user_name)  # Debugging: print the user name

    # Validate the user and method
    if user_name in valid_userid and request.method == 'POST':
        # Fetch the top 20 recommended products for the user
        top20_products = model.recommend_products(user_name)
        print(top20_products.head())  # Debugging: print the top rows of the recommendation data

        # Extract the top 5 products
        get_top5 = model.top5_products(top20_products)

        # Render the recommendations in the HTML page with a table and a message
        return render_template(
            'index.html',
            column_names=get_top5.columns.values,  # Pass column names for the table
            row_data=list(get_top5.values.tolist()),  # Pass row data for the table
            zip=zip,  # Allow zipping columns and rows in the template
            text='Recommended products'  # Pass text for user feedback
        )
    elif user_name not in valid_userid:
        # Render the home page with a "no recommendation" message for invalid users
        return render_template('index.html', text='No Recommendation found for the user')
    else:
        # Fallback rendering of the home page
        return render_template('index.html')

# Run the Flask application
if __name__ == '__main__':
    app.debug = False  # Disable debug mode in production
    app.run()  # Start the application
