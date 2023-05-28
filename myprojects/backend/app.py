from flask import Flask, render_template, request, redirect

app = Flask(__name__, template_folder="./templates")

users = []  # 사용자 정보를 저장할 리스트

@app.route('/')
def index():
    return 'Welcome to the Flask Registration App'

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        # 사용자 정보를 저장하거나 데이터베이스에 저장할 수 있습니다.
        users.append({'username': username, 'password': password, 'email': email})
        
        return redirect('/login')
    
    return render_template("register.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # 사용자 정보를 확인하거나 데이터베이스에서 검증할 수 있습니다.
        for user in users:
            if user['username'] == username and user['password'] == password:
                return f'Welcome, {username}!'
        
        return 'Invalid username or password'
    
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)