const express = require('express');
const multer = require('multer');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.post('/upload', upload.single('image'), (req, res) => {
    // 업로드된 이미지를 저장하고 해당 이미지의 경로를 생성
    const imageUrl = '/uploads/' + req.file.filename;
    // 이미지 저장 후 imageUrl을 클라이언트에게 응답
    res.json({ imageUrl });
});

app.get('/images', (req, res) => {
    // 실제로 서버에서 이미지를 가져와서 이미지 목록을 반환
    const imageUrls = ['image1.jpg', 'image2.jpg', 'image3.jpg'];
    res.json({ images: imageUrls }); ㅗㅅ
});

// 정적 파일 제공을 위한 미들웨어 설정
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// 서버 시작
app.listen(3000, () => {
    console.log('Server is running on port 3000');
});
