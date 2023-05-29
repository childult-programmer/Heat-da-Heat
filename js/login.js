const signup = document.getElementById("sign-up");
const signin = document.getElementById("sign-in");
const loginin = document.getElementById("login-in");
const loginup = document.getElementById("login-up");

signup.addEventListener("click", () => {
    loginin.classList.remove("block");
    loginup.classList.remove("none");

    loginin.classList.add("none");
    loginup.classList.add("block");
})

signin.addEventListener("click", () => {
    loginin.classList.remove("none");
    loginup.classList.remove("block");

    loginin.classList.add("block");
    loginup.classList.add("none");
})

const loginForm = document.querySelector('#login-in');
const createAccountForm = document.querySelector('#login-up');
const loginButton = document.querySelector('.login__register .login__button');
const createAccountButton = document.querySelector('.login__create .login__button');

loginForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const employeeId = loginForm.querySelector('input[name="employeeId"]').value;
    const password = loginForm.querySelector('input[name="password"]').value;

    const registeredUsers = JSON.parse(localStorage.getItem('registeredUsers')) || [];
    const currentUser = registeredUsers.find(user => user.employeeId === employeeId && user.password === password);

    if (currentUser) {
        localStorage.setItem('currentUser', JSON.stringify(currentUser));
        alert('로그인 성공');
        window.location.href = 'index.html';
    } else {
        alert('로그인 실패');
    }
});

createAccountForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const username = createAccountForm.querySelector('input[name="username"]').value;
    const employeeId = createAccountForm.querySelector('input[name="employeeId"]').value;
    const password = createAccountForm.querySelector('input[name="password"]').value;

    const registeredUsers = JSON.parse(localStorage.getItem('registeredUsers')) || [];

    const isUserAlreadyExist = registeredUsers.find(user => user.employeeId === employeeId);

    if (isUserAlreadyExist) {
        alert('이미 등록된 사용자입니다.');
    } else {
        const newUser = { username, employeeId, password };
        registeredUsers.push(newUser);
        localStorage.setItem('registeredUsers', JSON.stringify(registeredUsers));
        alert('회원가입 성공');
        window.location.href = 'index.html';
    }
});

loginButton.addEventListener('click', () => {
    loginForm.submit();
});

createAccountButton.addEventListener('click', () => {
    createAccountForm.submit();
});