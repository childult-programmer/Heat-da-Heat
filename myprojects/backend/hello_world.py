from flask import Flask #설치한 Flask 패키지에서 Flask 모듈을 import 하여 사용
app = Flask(__name__)  #플라스크를 생성하고 app 변수에 flask 초기화 하여 실행

@app.route('/') #사용자에게 ( ) 에 있는 경로를 안내한다
def hello_world(): #누군가가 위의 주소에 접근하면 함수 실행
    return 'hello world!'
    
if __name__ == '__main__':
    app.debug = True
    app.run()