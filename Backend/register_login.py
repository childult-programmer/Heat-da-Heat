from flask import Flask, render_template, request, redirect, jsonify, flash

app = Flask(__name__, template_folder="templates")
app.secret_key = '123'

users = []  # 사용자 정보를 저장할 리스트


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after_login')
def after_index():
    return render_template('index_login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        userID = request.form['Employee ID']
        password = request.form['Password']

        # 사용자 정보를 저장하거나 데이터베이스에 저장
        users.append(
            {'username': username, 'Employee ID': userID, 'Password': password})

        return redirect('/login')

    return render_template('create_account.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        userID = request.form['Employee ID']
        password = request.form['Password']

        # 사용자 정보를 확인하거나 데이터베이스에서의 정보 유무 확인
        for user in users:
            if user['Employee ID'] == userID and user['Password'] == password:
                return redirect('/after_login')

        flash('올바른 ID 또는 비밀번호를 입력하세요.')

    return render_template('login.html')


if __name__ == '__main__':
    app.run(debug=True)
