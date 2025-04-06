from flask import Flask, render_template

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')  # Serve the Home page

@app.route('/Lip', methods=['GET'])
def deaf():
    return render_template('Lip.html')  # Serve the Deaf page

@app.route('/About', methods=['GET'])
def mute():
    return render_template('About.html')  # Serve the Mute page

if __name__ == '__main__':
    app.run(debug=True)